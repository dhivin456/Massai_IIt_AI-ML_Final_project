-- Q1: WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS qualifying_spend
FROM orders WHERE status='completed' AND order_amount > 500
GROUP BY customer_id HAVING SUM(order_amount) > 8000 ORDER BY qualifying_spend DESC;

-- Q2: revenue by city and category
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id)
WHERE o.status='completed' GROUP BY c.city, p.category ORDER BY c.city, p.category;

-- Q3: products priced above catalogue average
SELECT product_id, product_name, price_inr FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products) ORDER BY price_inr DESC;

-- Q4: customers without completed orders
SELECT c.customer_id FROM customers c LEFT JOIN orders o
ON c.customer_id=o.customer_id AND o.status='completed'
WHERE o.order_id IS NULL ORDER BY c.customer_id;

-- Q5: two CTEs for customer totals and average total
WITH customer_totals AS (
 SELECT customer_id, SUM(order_amount) AS total_spend FROM orders
 WHERE status='completed' GROUP BY customer_id
), average_total AS (SELECT AVG(total_spend) AS avg_spend FROM customer_totals)
SELECT ct.customer_id, ct.total_spend FROM customer_totals ct CROSS JOIN average_total a
WHERE ct.total_spend > a.avg_spend ORDER BY ct.total_spend DESC;

-- Q6: rank completed spend within city
WITH totals AS (
 SELECT c.city, c.customer_id, SUM(o.order_amount) AS total_spend
 FROM customers c JOIN orders o USING(customer_id)
 WHERE o.status='completed' GROUP BY c.city, c.customer_id
)
SELECT city, customer_id, ROUND(total_spend,2) AS total_spend,
 RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
 DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,
 ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC, customer_id) AS row_num
FROM totals ORDER BY city, spend_rank, customer_id;

-- Q7: daily completed revenue and running total
WITH daily AS (
 SELECT substr(order_ts,1,10) AS day, SUM(order_amount) AS revenue
 FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)
)
SELECT day, ROUND(revenue,2) AS revenue,
 ROUND(SUM(revenue) OVER (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),2) AS cumulative_revenue
FROM daily ORDER BY day;
