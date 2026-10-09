-- At most one forecast per question, model version and forecast date.
--
-- The scheduler (Task 22) may run more than once on a day; this makes a
-- duplicate an error instead of a second row. Live and backtest forecasts
-- are kept apart, so backtesting a date doesn't collide with the live run.

CREATE UNIQUE INDEX uq_forecasts_question_model_date
    ON forecasts (question_id, model_version_id, forecast_date, is_backtest);
