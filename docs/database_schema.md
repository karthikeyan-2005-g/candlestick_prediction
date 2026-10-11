# Database schema

SQLAlchemy creates the `predictions` table using `python scripts/init_db.py`.

| Column | Type | Constraints / purpose |
| --- | --- | --- |
| `id` | Integer | Primary key |
| `symbol` | VARCHAR(24) | Optional uppercase market symbol |
| `timeframe` | VARCHAR(24) | Optional candle duration |
| `prediction` | VARCHAR(16) | Required archive class `Down` or `Up` (legacy directional labels also accepted) |
| `confidence` | Float | Required score in `[0, 1]` |
| `model_version` | VARCHAR(64) | Required checkpoint version |
| `image_sha256` | CHAR-like VARCHAR(64) | Required hash; original image is not stored |
| `created_at` | timezone-aware timestamp | Database generated |
| `actual_direction` | VARCHAR(16) | Nullable `Bullish` or `Bearish` |
| `observed_at` | timezone-aware timestamp | Nullable outcome observation time |

Check constraints enforce label values. Indexes support recent-history
ordering and symbol/time queries. There are no foreign keys because a
prediction is a standalone observation. SQLAlchemy's ORM and parameterized
queries are used for inserts, filtering, pagination, and updates.

For a disposable local development database, `db.create_all()` is sufficient.
It will not modify existing table definitions. For ongoing deployments install
Flask-Migrate/Alembic, initialize migration history, generate a migration for
each model change, review the generated SQL, back up PostgreSQL, and apply
migrations in a controlled release. Never drop production tables to apply a
schema update.

Example query:

```sql
SELECT id, symbol, timeframe, prediction, confidence, created_at,
       actual_direction, observed_at
FROM predictions
WHERE symbol = 'AAPL'
ORDER BY created_at DESC, id DESC
LIMIT 20;
```
