CREATE TABLE users (
    user_id             INT PRIMARY KEY,
    signup_date         DATE,
    acquisition_channel VARCHAR(50),
    age                 NUMERIC(5,1),
    city                VARCHAR(50)
);

CREATE TABLE activity (
    user_id       INT,
    event_date    DATE,
    session_count INT,
    feature_used  VARCHAR(50)
);

CREATE TABLE transactions (
    user_id          INT,
    transaction_date DATE,
    amount           BIGINT
);

CREATE TABLE status (
    user_id                  INT PRIMARY KEY,
    loan_count               INT,
    avg_transaction_amount   NUMERIC(15,2),
    days_since_last_activity INT,
    is_churn                 INT,
    churn_date               DATE
);

select*from activity;
select*from status;
select*from transactions;
select*from users;

 
SELECT
    COUNT(*) AS total_users,
    COUNT(*) FILTER (WHERE is_churn = 0) AS active_users,
    COUNT(*) FILTER (WHERE is_churn = 1) AS churn_users,
 
    ROUND(COUNT(*) FILTER (WHERE is_churn = 1)::NUMERIC
        / COUNT(*) * 100, 2) AS churn_rate_pct 
FROM status;

 ============================================================
 
SELECT
    u.acquisition_channel,
    COUNT(*) AS total_users,
    COUNT(*) FILTER (WHERE s.is_churn = 1) AS churn_users,
    COUNT(*) FILTER (WHERE s.is_churn = 0) AS active_users,
    ROUND( COUNT(*) FILTER (WHERE s.is_churn = 1)::NUMERIC
        / COUNT(*) * 100, 2) AS churn_rate_pct
 
FROM users u
JOIN status s USING (user_id)
 
GROUP BY u.acquisition_channel
ORDER BY churn_rate_pct DESC;

-- ============================================================
 
SELECT
    DATE_TRUNC('month', u.signup_date::DATE) AS signup_cohort,
    COUNT(*) AS total_users,
    COUNT(*) FILTER (WHERE s.is_churn = 1) AS churn_users,
    COUNT(*) FILTER (WHERE s.is_churn = 0) AS active_users,
    ROUND( COUNT(*) FILTER (WHERE s.is_churn = 1)::NUMERIC
        / COUNT(*) * 100, 2) AS churn_rate_pct
 
FROM users u
JOIN status s USING (user_id)
 
GROUP BY signup_cohort
ORDER BY signup_cohort;

SELECT
    DATE_TRUNC('month', u.signup_date::DATE) AS signup_cohort,
    u.acquisition_channel,
    COUNT(*) AS total_users,
    COUNT(*) FILTER (WHERE s.is_churn = 1) AS churn_users,
    ROUND( COUNT(*) FILTER (WHERE s.is_churn = 1)::NUMERIC
        / COUNT(*) * 100, 2) AS churn_rate_pct
 
FROM users u
JOIN status s USING (user_id)
 
GROUP BY signup_cohort, u.acquisition_channel
ORDER BY signup_cohort, churn_rate_pct DESC;


WITH last_activity_per_user AS (
    SELECT
        user_id,
        MAX(event_date::DATE) AS last_activity_date
    FROM activity
    GROUP BY user_id
)
 
SELECT
    u.user_id,
    u.acquisition_channel,
    u.city,
    la.last_activity_date,
 
    -- Berapa hari sudah tidak login
    CURRENT_DATE - la.last_activity_date AS days_inactive,
    s.is_churn
 
FROM last_activity_per_user la
JOIN users  u USING (user_id)
JOIN status s USING (user_id)
 
WHERE
    la.last_activity_date < CURRENT_DATE - INTERVAL '30 days'
    AND s.is_churn = 0
ORDER BY days_inactive DESC;


SELECT
    u.acquisition_channel,
    COUNT(*) AS total_users,
    COUNT(*) FILTER (WHERE s.is_churn = 0) AS retained_users,
    ROUND( COUNT(*) FILTER (WHERE s.is_churn = 0)::NUMERIC
        / COUNT(*) * 100
    , 2) AS retention_rate_pct
 
FROM users u
JOIN status s USING (user_id)
 
GROUP BY u.acquisition_channel
ORDER BY retention_rate_pct DESC

LIMIT 3;

SELECT
    u.acquisition_channel,
    COUNT(*) AS total_users,
    COUNT(*) FILTER (WHERE s.is_churn = 1) AS churn_users,
    ROUND( COUNT(*) FILTER (WHERE s.is_churn = 1)::NUMERIC
        / COUNT(*) * 100
    , 2) AS churn_rate_pct
 
FROM users u
JOIN status s USING (user_id)
 
GROUP BY u.acquisition_channel
ORDER BY churn_rate_pct DESC
LIMIT 3;

SELECT
    u.acquisition_channel,
    COUNT(*)                                         AS total_users,
    COUNT(*) FILTER (WHERE s.is_churn = 0)           AS active_users,
    COUNT(*) FILTER (WHERE s.is_churn = 1)           AS churn_users,
    ROUND(
        COUNT(*) FILTER (WHERE s.is_churn = 0)::NUMERIC
        / COUNT(*) * 100
    , 2)                                             AS retention_rate_pct,
    ROUND(
        COUNT(*) FILTER (WHERE s.is_churn = 1)::NUMERIC
        / COUNT(*) * 100
    , 2)                                             AS churn_rate_pct
 
FROM users u
JOIN status s USING (user_id)
 
GROUP BY u.acquisition_channel
ORDER BY retention_rate_pct DESC;
 