# File: RLTrader.py
import torch, os, threading, random, json
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

import _classes.Constants as CONSTANTS
from _classes.RLEnvironment import RLTradingEnvironment, N_BLENDS, decode_action, REEVAL_INTERVAL
from _classes.RLNetwork import MomentumActorCritic
from _classes.RLTrainer import PPOTrainer
from _classes.Trading import TradingModel, TradeModelParams
from _classes.Selection import StockPicker
from _classes.TickerLists import TickerLists
from _classes.Prices import PricingData

VERSION    = "V1.3"
STATE_DIM  = 33   # 14 market + 7 universe + 7 portfolio + 3 calendar + 2 last-action
ACTION_DIM = N_BLENDS * 2             # 16: 8 blends × 2 (hold/rebalance)

ALL_PERIODS = [
	('1/1/1980', 3), ('1/1/1983', 3), ('1/1/1986', 3), ('1/1/1989', 3), ('1/1/1992', 3),
	('1/1/1995', 3), ('1/1/1998', 3), ('1/1/2001', 3), ('1/1/2004', 3), ('1/1/2007', 3),
	('1/1/2010', 3), ('1/1/2013', 3), ('1/1/2016', 3), ('1/1/2019', 3), ('1/1/2022', 3),
]

COL_ORDER = [
	"test_start_year", "test_end_year",
	"final_value", "total_return_pct", "cagr_pct", "sharpe_ratio", "max_drawdown_pct",
	"bm_final_value", "bm_total_return_pct", "bm_cagr_pct", "bm_sharpe", "bm_max_drawdown_pct",
	"n_steps", "n_rebalances", "avg_reward", "total_reward", "model_path",
]

def make_params(start: str, years: int, model_name: str = None) -> TradeModelParams:
	p = TradeModelParams()
	p.startDate = start
	p.durationInYears = years
	p.portfolioSize = 100_000
	p.reEvaluationInterval = 5
	p.pickHistoryWindow = 28
	p.rateLimitTransactions = True
	p.saveTradeHistory = False
	if model_name:
		p.modelName = model_name
	return p


def train_one_period(model_path: str, period: tuple, picker: StockPicker, n_passes: int = 50, fold_label: str = None):
	"""
	Train (or continue training) a fold's model on a single period.
	Loads from model_path if it exists, otherwise initialises a fresh model.
	Saves the updated checkpoint back to model_path when done.
	fold_label identifies which CV fold owns this model (e.g. "test1983").
	"""
	start, years = period
	start_year = pd.Timestamp(start).year
	label = fold_label or f"train{start_year}"

	model = MomentumActorCritic(state_dim=STATE_DIM, action_dim=ACTION_DIM)
	checkpoint = None
	if os.path.exists(model_path):
		checkpoint = torch.load(model_path, weights_only=True)
		model.load_state_dict(checkpoint["model_state_dict"])

	trainer = PPOTrainer(model, lr=3e-4)
	if checkpoint and "optimizer_state_dict" in checkpoint:
		trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

	total_ep = checkpoint.get("episode", 0) if checkpoint else 0

	for pass_num in range(n_passes):
		params = make_params(start, years, f"RL_{label}_on{start_year}")
		env    = RLTradingEnvironment(params, picker)
		if pass_num == 0:
			bc = trainer.bc_warmup(env)
			print(f"    BC warmup: loss={bc['bc_loss']:.4f}  samples={bc['bc_samples']}")
		# Linear entropy decay per period: 0.10 (explore) → 0.01 (exploit)
		progress = pass_num / max(n_passes - 1, 1)
		trainer.entropy_coef = 0.10 * (1.0 - 0.9 * progress)
		# Cosine LR schedule per period: 3e-4 → 1e-5
		lr = 1e-5 + 0.5 * (3e-4 - 1e-5) * (1.0 + float(np.cos(np.pi * progress)))
		for param_group in trainer.optimizer.param_groups:
			param_group['lr'] = lr
		stats  = trainer.run_episode(env)
		total_ep += 1
		print(f"    pass {pass_num+1}/{n_passes}  ep={total_ep:3d} | "
			  f"Return: {stats['episode_return']:+.3f} | "
			  f"Final: ${stats['final_portfolio_value']:,.0f} | "
			  f"Entropy: {stats['entropy']:.3f} | "
			  f"PolicyLoss: {stats['policy_loss']:.4f}")

	params_dict = {k: str(v) if isinstance(v, pd.Timestamp) else v
				   for k, v in vars(params).items()}
	torch.save({
		"model_state_dict":     model.state_dict(),
		"optimizer_state_dict": trainer.optimizer.state_dict(),
		"episode":              total_ep,
		"params":               params_dict,
	}, model_path)

def run_rl_trader(model_path: str, params: TradeModelParams, picker: StockPicker) -> dict:
	"""Run the trained RL model in inference mode. Returns a metrics dict."""
	model = MomentumActorCritic(state_dim=STATE_DIM, action_dim=ACTION_DIM)
	model.load_state_dict(torch.load(model_path, weights_only=True)["model_state_dict"])
	model.eval()

	env   = RLTradingEnvironment(params, picker)
	state = env.reset()
	done  = False
	action_log = []

	while not done:
		state_t = torch.FloatTensor(state).unsqueeze(0)
		with torch.no_grad():
			action, _, _, value = model.get_action(state_t, deterministic=True)
		blend_key, stock_count, rebalance_now = decode_action(action.item())
		next_state, reward, done, info = env.step(action.item())
		action_log.append({
			"date":            env.tm.currentDate,
			"blend":           blend_key,
			"stock_count":     stock_count,
			"rebalance":       rebalance_now,
			"reward":          reward,
			"value":           info["portfolio_value"],
			"estimated_value": value.item(),
		})
		state = next_state

	final_value = env.tm.CloseModel(params)
	start_value = params.portfolioSize
	years = params.durationInYears
	cagr  = ((final_value / start_value) ** (1.0 / years) - 1) * 100 if years > 0 else 0.0

	log_df = pd.DataFrame(action_log)
	steps_per_year = 252 / REEVAL_INTERVAL
	if len(log_df) > 1:
		vals      = log_df["value"].values
		step_rets = np.diff(vals) / np.maximum(vals[:-1], 1e-8)
		sharpe    = float(step_rets.mean() / (step_rets.std() + 1e-8) * np.sqrt(steps_per_year))
		peak      = np.maximum.accumulate(vals)
		max_dd    = float((vals - peak).min() / np.maximum(peak.max(), 1e-8) * 100)
	else:
		sharpe = 0.0
		max_dd = 0.0
	return {
		"final_value":      final_value,
		"total_return_pct": (final_value / start_value - 1) * 100,
		"cagr_pct":         cagr,
		"sharpe_ratio":     sharpe,
		"max_drawdown_pct": max_dd,
		"n_steps":          len(action_log),
		"n_rebalances":     int(log_df["rebalance"].sum()) if not log_df.empty else 0,
		"avg_reward":       float(log_df["reward"].mean()) if not log_df.empty else 0.0,
		"total_reward":     float(log_df["reward"].sum())  if not log_df.empty else 0.0,
	}

_inx_prices_cache: pd.Series = None  # Pre-loaded in main thread; thread-safe for reads

def _preload_inx(all_periods: list):
	"""Load the full .INX price history once in the main thread before spawning workers.
	Avoids concurrent SQL/download attempts that fail silently under threading."""
	global _inx_prices_cache
	full_start = all_periods[0][0]
	last_start, last_years = all_periods[-1]
	full_end = str((pd.Timestamp(last_start) + pd.DateOffset(years=last_years + 1)).date())
	p = PricingData('.INX')
	if p.LoadHistory(requestedStartDate=full_start, requestedEndDate=full_end):
		_inx_prices_cache = p.historicalPrices['Close']
		print(f"  .INX loaded: {_inx_prices_cache.index[0].date()} → {_inx_prices_cache.index[-1].date()}  ({len(_inx_prices_cache)} days)")
	else:
		print("  Warning: could not load .INX prices — benchmark metrics will be 0")


def run_benchmark(params: TradeModelParams) -> dict:
	"""S&P 500 buy-and-hold benchmark computed from pre-loaded .INX price cache."""
	if _inx_prices_cache is None:
		return {"bm_final_value": params.portfolioSize, "bm_total_return_pct": 0.0,
				"bm_cagr_pct": 0.0, "bm_sharpe": 0.0, "bm_max_drawdown_pct": 0.0}

	start = pd.Timestamp(params.startDate)
	end   = start + pd.DateOffset(years=params.durationInYears)
	prices = _inx_prices_cache[
		(_inx_prices_cache.index >= start) & (_inx_prices_cache.index <= end)
	]
	if len(prices) < 2:
		return {"bm_final_value": params.portfolioSize, "bm_total_return_pct": 0.0,
				"bm_cagr_pct": 0.0, "bm_sharpe": 0.0, "bm_max_drawdown_pct": 0.0}

	start_value = params.portfolioSize
	port_vals   = (prices.values / prices.values[0]) * start_value
	final_value = float(port_vals[-1])
	years       = params.durationInYears
	cagr        = ((final_value / start_value) ** (1.0 / years) - 1) * 100 if years > 0 else 0.0

	steps_per_year = 252 / REEVAL_INTERVAL
	sampled = port_vals[::REEVAL_INTERVAL]
	if len(sampled) > 1:
		step_rets = np.diff(sampled) / np.maximum(sampled[:-1], 1e-8)
		sharpe    = float(step_rets.mean() / (step_rets.std() + 1e-8) * np.sqrt(steps_per_year))
		peak      = np.maximum.accumulate(sampled)
		max_dd    = float((sampled - peak).min() / np.maximum(peak.max(), 1e-8) * 100)
	else:
		sharpe, max_dd = 0.0, 0.0

	return {
		"bm_final_value":      final_value,
		"bm_total_return_pct": (final_value / start_value - 1) * 100,
		"bm_cagr_pct":         cagr,
		"bm_sharpe":           sharpe,
		"bm_max_drawdown_pct": max_dd,
	}

_print_lock = threading.Lock()
_state_lock = threading.Lock()

STATE_PATH = "data/rl_training_state.json"

def _load_state() -> dict:
	"""Load persisted training state, or return a blank slate on first run / corruption."""
	try:
		with open(STATE_PATH) as f:
			return json.load(f)
	except (FileNotFoundError, json.JSONDecodeError):
		return {"completed_folds": {}, "trained_periods": {}}

def _persist_state(state: dict):
	"""Atomically write state via temp-file rename so a crash never corrupts the file."""
	tmp = STATE_PATH + ".tmp"
	with open(tmp, "w") as f:
		json.dump(state, f, indent=2)
	os.replace(tmp, STATE_PATH)

def _save_results(fold_results: list, results_path: str, n_total: int):
	df  = pd.DataFrame(fold_results).sort_values("test_start_year")[COL_ORDER]
	avg = df.select_dtypes(include="number").mean()
	avg_row = {c: avg.get(c, "") for c in COL_ORDER}
	avg_row["test_start_year"] = "AVERAGE"
	avg_row["test_end_year"]   = ""
	avg_row["model_path"]      = ""
	out_df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
	out_df.to_csv(results_path, index=False)
	print(f"  Saved to {results_path}  ({len(fold_results)}/{n_total} folds complete)")
	return avg

def _run_fold(fold_idx: int, all_periods: list, state: dict) -> dict:
	"""
	Run one complete CV fold in its own thread.
	Creates a private StockPicker so there is no shared mutable state between threads.
	Skips training periods already completed in a prior run (crash-safe resume).
	"""
	test_start, test_years = all_periods[fold_idx]
	test_year  = pd.Timestamp(test_start).year
	model_path = f"data/models/rl_trader_{VERSION}_test_year{test_year}.pt"

	full_start = all_periods[0][0]
	last_start, last_years = all_periods[-1]
	full_end = pd.Timestamp(last_start) + pd.DateOffset(years=last_years)

	with _print_lock:
		print(f"[fold test={test_year}] Starting — own picker {full_start} → {full_end.date()}")
	picker = StockPicker(startDate=full_start, endDate=full_end, pickHistoryWindow=28)

	# ── Training ─────────────────────────────────────────────────────────────
	with _state_lock:
		done_periods = set(state["trained_periods"].get(str(test_year), []))

	train_indices = [i for i in range(len(all_periods)) if i != fold_idx]
	for i in train_indices:
		train_start, train_years = all_periods[i]
		train_year = pd.Timestamp(train_start).year
		if train_start in done_periods:
			with _print_lock:
				print(f"[fold test={test_year}] Skipping {train_year} (already trained)")
			continue
		tickers = TickerLists.GetTickerListSQL(year=train_year, month=1)
		with _print_lock:
			print(f"[fold test={test_year}] Training on {train_year}  ({len(tickers)} tickers)")
		picker.AlignToList(tickers)
		train_one_period(model_path, (train_start, train_years), picker, fold_label=f"test{test_year}")
		with _state_lock:
			state["trained_periods"].setdefault(str(test_year), []).append(train_start)
			_persist_state(state)

	# ── Testing ───────────────────────────────────────────────────────────────
	tickers = TickerLists.GetTickerListSQL(year=test_year, month=1)
	with _print_lock:
		print(f"[fold test={test_year}] Testing  ({len(tickers)} tickers)")
	picker.AlignToList(tickers)

	test_params = make_params(test_start, test_years, f"RL_TraderTest_year={test_year}")
	test_params.saveTradeHistory = True
	bm_params   = make_params(test_start, test_years)
	metrics     = run_rl_trader(model_path, test_params, picker)
	bm_metrics  = run_benchmark(bm_params)
	metrics.update(bm_metrics)
	metrics["test_start_year"] = test_year
	metrics["test_end_year"]   = test_year + test_years - 1
	metrics["model_path"]      = model_path

	with _state_lock:
		state["completed_folds"][str(test_year)] = metrics
		state["trained_periods"].pop(str(test_year), None)  # no longer needed
		_persist_state(state)

	with _print_lock:
		print(f"[fold test={test_year}] Done | "
			  f"RL: {metrics['total_return_pct']:+.1f}% / {metrics['cagr_pct']:+.1f}% CAGR  "
			  f"BM: {metrics['bm_total_return_pct']:+.1f}% / {metrics['bm_cagr_pct']:+.1f}% CAGR")
	return metrics

def train_full_history_model(
	n_passes: int = 90,
	n_sweeps: int = 3,
	model_path: str = None,
) -> str:
	"""
	Train a model on all 15 periods with no holdout fold, for use in run_full_history_test.
	CV fold models each hold out one period, so none is optimal for a continuous 45-year run.
	Uses historically-accurate ticker lists for each 3-year period.
	Multiple sweeps revisit all periods so later periods reinforce earlier learning.
	"""
	if model_path is None:
		model_path = f"data/models/rl_trader_{VERSION}_full_history.pt"
	os.makedirs("data/models", exist_ok=True)

	full_start = ALL_PERIODS[0][0]
	last_start, last_years = ALL_PERIODS[-1]
	full_end = pd.Timestamp(last_start) + pd.DateOffset(years=last_years)

	passes_per_sweep = max(n_passes // n_sweeps, 1)
	print(f"\nTraining full-history model: {n_sweeps} sweeps × {passes_per_sweep} passes "
		  f"across {len(ALL_PERIODS)} periods ...")
	picker = StockPicker(startDate=full_start, endDate=str(full_end.date()), pickHistoryWindow=28)

	for sweep in range(n_sweeps):
		print(f"\n  Sweep {sweep + 1}/{n_sweeps}")
		for start, years in ALL_PERIODS:
			train_year = pd.Timestamp(start).year
			tickers = TickerLists.GetTickerListSQL(year=train_year, month=1)
			print(f"    Period {train_year}: {len(tickers)} tickers")
			picker.AlignToList(tickers)
			train_one_period(model_path, (start, years), picker, n_passes=passes_per_sweep, fold_label=f"full_s{sweep+1}")

	print(f"  Full-history model saved to {model_path}")
	return model_path


def run_full_history_test(
	model_path: str = None,
	start: str = '1/1/1980',
	years: int = 45,
	results_path: str = "data/rl_full_history_test.csv",
):
	"""Run a single trained model across the full 45-year history for comparison with AlphaTrader."""
	if model_path is None:
		full_history_path = f"data/models/rl_trader_{VERSION}_full_history.pt"
		fallback_path     = f"data/models/rl_trader_{VERSION}_test_year2022.pt"
		model_path = full_history_path if os.path.exists(full_history_path) else fallback_path
	if not os.path.exists(model_path):
		print(f"  Model not found: {model_path}")
		return

	print(f"\nFull-history test: {model_path}")
	print(f"  Period: {start} → {years} years")

	end_date = pd.Timestamp(start) + pd.DateOffset(years=years)

	# Union of historically-accurate ticker lists gives the picker access to all stocks
	# that were S&P components in each era, not just the 2022 list.
	all_tickers: set = set()
	for period_start, _ in ALL_PERIODS:
		period_year = pd.Timestamp(period_start).year
		all_tickers.update(TickerLists.GetTickerListSQL(year=period_year, month=1))
	print(f"  Loading picker ({len(all_tickers)} tickers across all periods) ...")
	picker = StockPicker(startDate=start, endDate=str(end_date.date()), pickHistoryWindow=28)
	picker.AlignToList(list(all_tickers))

	params = make_params(start, years, "RL_FullHistory_Test")
	params.saveTradeHistory = True
	metrics = run_rl_trader(model_path, params, picker)

	if _inx_prices_cache is None:
		_preload_inx([('1/1/1980', 1), (str(end_date.year - 1) + '/1/1', 1)])
	bm = run_benchmark(make_params(start, years))
	metrics.update(bm)

	print(f"\n  RL:  Return={metrics['total_return_pct']:+.1f}%  CAGR={metrics['cagr_pct']:+.1f}%  "
		  f"Sharpe={metrics['sharpe_ratio']:.2f}  MaxDD={metrics['max_drawdown_pct']:.1f}%")
	print(f"  BM:  Return={metrics['bm_total_return_pct']:+.1f}%  CAGR={metrics['bm_cagr_pct']:+.1f}%  "
		  f"Sharpe={metrics['bm_sharpe']:.2f}  MaxDD={metrics['bm_max_drawdown_pct']:.1f}%")

	pd.DataFrame([metrics]).to_csv(results_path, index=False)
	print(f"  Saved to {results_path}")
	return metrics


def cross_validate(all_periods: list, results_path: str = "data/rl_cross_validation.csv"):
	"""
	Leave-one-out cross-validation — one thread per fold, all running in parallel.
	Crash-safe: completed folds are recorded in rl_training_state.json and skipped on
	restart. Partial folds resume from the last completed training period.
	"""
	os.makedirs("data/models", exist_ok=True)
	os.makedirs("data", exist_ok=True)
	_preload_inx(all_periods)  # Load .INX once before spawning threads

	state = _load_state()
	completed_years = set(state["completed_folds"].keys())

	n = len(all_periods)
	fold_results = list(state["completed_folds"].values())
	pending = [i for i, (s, _) in enumerate(all_periods)
			   if str(pd.Timestamp(s).year) not in completed_years]

	if fold_results:
		print(f"Resuming: {len(fold_results)}/{n} folds already complete.")
		_save_results(fold_results, results_path, n)

	if not pending:
		print("All folds already complete.")
		return

	print(f"Running {len(pending)} remaining folds on {len(pending)} threads ...")

	with ThreadPoolExecutor(max_workers=len(pending)) as executor:
		futures = {executor.submit(_run_fold, i, all_periods, state): i for i in pending}
		for future in as_completed(futures):
			metrics = future.result()
			fold_results.append(metrics)
			_save_results(fold_results, results_path, n)

	avg = _save_results(fold_results, results_path, n)
	print(f"\n{'='*60}")
	print(f"Cross-validation complete. Results saved to {results_path}")
	print(f"  Avg Return: {avg['total_return_pct']:+.1f}%  |  Avg CAGR: {avg['cagr_pct']:+.1f}%")

if __name__ == "__main__":
	#cross_validate(ALL_PERIODS)
	train_full_history_model()
	run_full_history_test()
