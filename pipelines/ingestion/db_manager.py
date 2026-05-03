"""
数据库管理器
处理PostgreSQL和DuckDB连接和数据操作
"""

import psycopg2
import psycopg2.extras
import duckdb
import pandas as pd
from typing import List, Dict, Optional, Any
from contextlib import contextmanager
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """数据库管理器"""

    def __init__(self, postgres_config: Dict[str, str], duckdb_path: str):
        """
        初始化数据库管理器

        Args:
            postgres_config: PostgreSQL配置
            duckdb_path: DuckDB数据库路径
        """
        self.postgres_config = postgres_config
        self.duckdb_path = duckdb_path
        self.postgres_conn = None
        self.duckdb_conn = None

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    def connect(self):
        """连接数据库"""
        try:
            # 连接PostgreSQL
            self.postgres_conn = psycopg2.connect(**self.postgres_config)
            logger.info("成功连接到PostgreSQL数据库")

            # 连接DuckDB
            self.duckdb_conn = duckdb.connect(self.duckdb_path)
            logger.info(f"成功连接到DuckDB数据库: {self.duckdb_path}")

        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.postgres_conn:
            self.postgres_conn.close()
            logger.info("关闭PostgreSQL连接")

        if self.duckdb_conn:
            self.duckdb_conn.close()
            logger.info("关闭DuckDB连接")

    @contextmanager
    def get_postgres_cursor(self):
        """获取PostgreSQL游标的上下文管理器"""
        cursor = None
        try:
            cursor = self.postgres_conn.cursor()
            yield cursor
        finally:
            if cursor:
                cursor.close()

    def execute_postgres(self, sql: str, params: tuple = None) -> Optional[List[Dict]]:
        """执行PostgreSQL查询"""
        try:
            with self.get_postgres_cursor() as cursor:
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)

                # 如果是SELECT查询，返回结果
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                    return results
                else:
                    # 对于非SELECT查询，提交事务
                    self.postgres_conn.commit()
                    return None

        except Exception as e:
            logger.error(f"PostgreSQL查询执行失败: {e}")
            self.postgres_conn.rollback()
            raise

    def insert_dataframe_postgres(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        """将DataFrame插入PostgreSQL"""
        try:
            # 将DataFrame转换为记录列表
            records = df.to_records(index=False)
            columns = df.columns.tolist()

            # 构建INSERT语句
            placeholders = ', '.join(['%s'] * len(columns))
            column_names = ', '.join(columns)
            sql = f"INSERT INTO {table_name} ({column_names}) VALUES ({placeholders})"

            # 批量插入
            with self.get_postgres_cursor() as cursor:
                psycopg2.extras.execute_batch(cursor, sql, [tuple(record) for record in records])

            self.postgres_conn.commit()
            logger.info(f"成功插入 {len(records)} 条记录到 {table_name}")

        except Exception as e:
            logger.error(f"插入数据到 {table_name} 失败: {e}")
            self.postgres_conn.rollback()
            raise

    def create_analytical_tables_duckdb(self) -> bool:
        """Create analytical views in DuckDB backed by PostgreSQL via postgres_scanner.

        Returns True if the view was created successfully, False otherwise.
        """
        try:
            self.duckdb_conn.execute("INSTALL postgres_scanner")
            self.duckdb_conn.execute("LOAD postgres_scanner")

            pg = self.postgres_config
            pg_conn_str = (
                f"host={pg['host']} port={pg['port']} dbname={pg['database']} "
                f"user={pg['user']} password={pg['password']}"
            )
            self.duckdb_conn.execute(f"ATTACH '{pg_conn_str}' AS pg (TYPE POSTGRES)")

            self.duckdb_conn.execute("""
                CREATE OR REPLACE VIEW company_daily_features AS
                SELECT
                    dmd.ticker,
                    dmd.date,
                    dmd.close_price,
                    dmd.volume,
                    dmd.volatility_5d,
                    dmd.volatility_20d,
                    dmd.returns_1d,
                    dmd.returns_5d,
                    dmd.returns_20d,
                    dmd.volume_ma_5d,
                    dmd.volume_ma_20d,
                    COUNT(DISTINCT ni.news_id) as news_count_7d,
                    AVG(COALESCE(lns.credit_risk_score, 0)) as avg_credit_risk_7d,
                    MAX(COALESCE(lns.credit_risk_score, 0)) as max_credit_risk_7d,
                    COUNT(DISTINCT CASE WHEN lns.credit_risk_score > 70 THEN lns.news_id END) as high_risk_news_7d,
                    ff.debt_to_assets,
                    ff.current_ratio,
                    ff.gross_margin,
                    ff.revenue_growth_yoy,
                    rl.abnormal_negative_return_1d,
                    rl.abnormal_volume_spike_1d,
                    rl.volatility_jump_5d,
                    rl.credit_proxy_widening_5d
                FROM pg.daily_market_data dmd
                LEFT JOIN pg.news_items ni ON dmd.ticker = ni.ticker
                    AND ni.published_at BETWEEN dmd.date - INTERVAL '7 days' AND dmd.date
                LEFT JOIN pg.llm_news_signals lns ON ni.news_id = lns.news_id
                LEFT JOIN pg.financial_fundamentals ff ON dmd.ticker = ff.ticker
                    AND ff.report_date <= dmd.date
                    AND ff.report_date >= dmd.date - INTERVAL '90 days'
                LEFT JOIN pg.risk_labels rl ON dmd.ticker = rl.ticker AND dmd.date = rl.date
                GROUP BY dmd.ticker, dmd.date, dmd.close_price, dmd.volume, dmd.volatility_5d,
                         dmd.volatility_20d, dmd.returns_1d, dmd.returns_5d, dmd.returns_20d,
                         dmd.volume_ma_5d, dmd.volume_ma_20d,
                         ff.debt_to_assets, ff.current_ratio, ff.gross_margin, ff.revenue_growth_yoy,
                         rl.abnormal_negative_return_1d, rl.abnormal_volume_spike_1d,
                         rl.volatility_jump_5d, rl.credit_proxy_widening_5d
            """)

            logger.info("DuckDB analytical view created successfully")
            return True

        except Exception as e:
            logger.warning(f"DuckDB analytical view creation skipped (PostgreSQL may not be available): {e}")
            return False

    def save_to_analytical_store(self, df: pd.DataFrame, table_name: str):
        """保存数据到DuckDB分析存储"""
        try:
            # 注册DataFrame到DuckDB
            self.duckdb_conn.register('temp_df', df)

            # 创建或替换表
            self.duckdb_conn.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS
                SELECT * FROM temp_df
            """)

            logger.info(f"成功保存 {len(df)} 条记录到DuckDB表 {table_name}")

        except Exception as e:
            logger.error(f"保存到DuckDB失败: {e}")
            raise

    def load_from_analytical_store(self, table_name: str, where_clause: str = None) -> pd.DataFrame:
        """从DuckDB分析存储加载数据"""
        try:
            query = f"SELECT * FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"

            df = self.duckdb_conn.execute(query).df()
            logger.info(f"从DuckDB加载 {len(df)} 条记录从 {table_name}")

            return df

        except Exception as e:
            logger.error(f"从DuckDB加载失败: {e}")
            raise

    def create_postgres_tables(self, schema_file: str):
        """从SQL文件创建PostgreSQL表"""
        try:
            with open(schema_file, 'r') as f:
                schema_sql = f.read()

            with self.get_postgres_cursor() as cursor:
                cursor.execute(schema_sql)

            self.postgres_conn.commit()
            logger.info(f"成功执行schema文件: {schema_file}")

        except Exception as e:
            logger.error(f"创建表失败: {e}")
            self.postgres_conn.rollback()
            raise

    def get_company_list(self) -> List[Dict]:
        """获取公司列表"""
        sql = """
            SELECT ticker, company_name, sector, industry, exchange, market_cap
            FROM companies
            ORDER BY market_cap DESC
        """
        return self.execute_postgres(sql) or []

    def insert_companies(self, companies: List[Dict]):
        """Insert company records (append-only, may fail on duplicates)."""
        df = pd.DataFrame(companies)
        self.insert_dataframe_postgres(df, 'companies')

    def upsert_companies(self, companies: List[Dict]):
        """Insert or update company records by ticker."""
        with self.get_postgres_cursor() as cursor:
            for c in companies:
                cursor.execute("""
                    INSERT INTO companies (ticker, company_name, sector, industry, exchange, market_cap, country, founded_year)
                    VALUES (%(ticker)s, %(company_name)s, %(sector)s, %(industry)s, %(exchange)s, %(market_cap)s, %(country)s, %(founded_year)s)
                    ON CONFLICT (ticker) DO UPDATE SET
                        company_name = EXCLUDED.company_name,
                        sector = EXCLUDED.sector,
                        industry = EXCLUDED.industry,
                        exchange = EXCLUDED.exchange,
                        market_cap = EXCLUDED.market_cap,
                        country = EXCLUDED.country,
                        founded_year = EXCLUDED.founded_year
                """, {
                    'ticker': c.get('ticker', ''),
                    'company_name': c.get('company_name', ''),
                    'sector': c.get('sector'),
                    'industry': c.get('industry'),
                    'exchange': c.get('exchange'),
                    'market_cap': c.get('market_cap'),
                    'country': c.get('country'),
                    'founded_year': c.get('founded_year'),
                })
            self.postgres_conn.commit()
        logger.info(f"Upserted {len(companies)} companies")

    def get_pending_news(self, limit: int = 1000) -> List[Dict]:
        """获取待处理的新闻"""
        sql = """
            SELECT news_id, ticker, title, body, source, url, published_at
            FROM news_items
            WHERE is_processed = FALSE
            ORDER BY published_at DESC
            LIMIT %s
        """
        return self.execute_postgres(sql, (limit,)) or []

    def mark_news_as_processed(self, news_ids: List[int]):
        """标记新闻为已处理"""
        sql = """
            UPDATE news_items
            SET is_processed = TRUE
            WHERE news_id = ANY(%s)
        """
        self.execute_postgres(sql, (news_ids,))

# 配置类
class DatabaseConfig:
    """数据库配置"""

    @staticmethod
    def get_default_postgres_config() -> Dict[str, str]:
        """获取默认PostgreSQL配置"""
        return {
            'host': 'localhost',
            'port': '5432',
            'database': 'creditmosaic',
            'user': 'postgres',
            'password': 'password'
        }

    @staticmethod
    def get_default_duckdb_path() -> str:
        """获取默认DuckDB路径"""
        return 'creditmosaic_analytical.db'

# 使用示例
if __name__ == "__main__":
    # 创建数据库管理器
    postgres_config = DatabaseConfig.get_default_postgres_config()
    duckdb_path = DatabaseConfig.get_default_duckdb_path()

    with DatabaseManager(postgres_config, duckdb_path) as db:
        # 创建表
        db.create_postgres_tables('db/schema.sql')

        # 插入示例公司数据
        companies = [
            {'ticker': 'AAPL', 'company_name': 'Apple Inc.', 'sector': 'Technology', 'market_cap': 2800000000000},
            {'ticker': 'MSFT', 'company_name': 'Microsoft Corporation', 'sector': 'Technology', 'market_cap': 2200000000000},
            {'ticker': 'GOOGL', 'company_name': 'Alphabet Inc.', 'sector': 'Technology', 'market_cap': 1800000000000},
        ]
        db.insert_companies(companies)

        # 获取公司列表
        company_list = db.get_company_list()
        print(f"公司数量: {len(company_list)}")