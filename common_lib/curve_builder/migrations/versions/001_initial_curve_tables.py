"""Initial curve tables migration

Revision ID: 001
Revises: 
Create Date: 2025-08-01

This migration creates the initial tables for curve data storage,
including historical curve data and PCA factor archives.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create curve-related tables for historical data."""
    
    # Create curve_mid_5s_historical hypertable for long-term storage
    op.execute("""
        CREATE TABLE IF NOT EXISTS market_data.curve_mid_5s_historical (
            ts TIMESTAMPTZ NOT NULL,
            cl1 DOUBLE PRECISION,
            cl2 DOUBLE PRECISION,
            cl3 DOUBLE PRECISION,
            cl4 DOUBLE PRECISION,
            cl5 DOUBLE PRECISION,
            cl6 DOUBLE PRECISION,
            cl7 DOUBLE PRECISION,
            cl8 DOUBLE PRECISION,
            total_volume BIGINT,
            contracts_available INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (ts)
        );
        
        -- Convert to hypertable if TimescaleDB is available
        DO $$ 
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
                PERFORM create_hypertable('market_data.curve_mid_5s_historical', 'ts', 
                    chunk_time_interval => INTERVAL '1 month',
                    if_not_exists => TRUE);
                
                -- Enable compression
                ALTER TABLE market_data.curve_mid_5s_historical 
                SET (timescaledb.compress = true);
                
                -- Add compression policy (compress chunks older than 3 months)
                PERFORM add_compression_policy('market_data.curve_mid_5s_historical', 
                    INTERVAL '3 months',
                    if_not_exists => TRUE);
            END IF;
        END $$;
        
        -- Create index for efficient queries
        CREATE INDEX IF NOT EXISTS idx_curve_mid_5s_historical_ts 
        ON market_data.curve_mid_5s_historical (ts DESC);
    """)
    
    # Create PCA factor archive table
    op.execute("""
        CREATE TABLE IF NOT EXISTS market_data.pca_factors_archive (
            id SERIAL PRIMARY KEY,
            ts TIMESTAMPTZ NOT NULL,
            factor_type VARCHAR(20) NOT NULL, -- 'price' or 'carry'
            bar_freq VARCHAR(10),
            window_bars INTEGER,
            factor_idx INTEGER NOT NULL,
            score DOUBLE PRECISION NOT NULL,
            exp_var DOUBLE PRECISION NOT NULL,
            loadings DOUBLE PRECISION[],
            quality_score DOUBLE PRECISION,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            archived_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(ts, factor_type, bar_freq, window_bars, factor_idx)
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_pca_archive_ts 
        ON market_data.pca_factors_archive (ts DESC);
        
        CREATE INDEX IF NOT EXISTS idx_pca_archive_type 
        ON market_data.pca_factors_archive (factor_type, ts DESC);
    """)
    
    # Create curve versions tracking table
    op.execute("""
        CREATE TABLE IF NOT EXISTS market_data.curve_versions (
            version_id SERIAL PRIMARY KEY,
            version_tag VARCHAR(50) NOT NULL UNIQUE,
            description TEXT,
            sql_hash VARCHAR(64),
            python_hash VARCHAR(64),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            created_by VARCHAR(100),
            is_active BOOLEAN DEFAULT TRUE
        );
        
        -- Insert initial version
        INSERT INTO market_data.curve_versions (version_tag, description, created_by)
        VALUES ('v1.0.0', 'Initial curve calculation version', 'curve_builder')
        ON CONFLICT (version_tag) DO NOTHING;
    """)
    
    # Create archive manifest for S3 storage
    op.execute("""
        CREATE TABLE IF NOT EXISTS market_data.archive_manifest (
            manifest_id SERIAL PRIMARY KEY,
            s3_key VARCHAR(500) NOT NULL UNIQUE,
            data_type VARCHAR(50) NOT NULL, -- 'curve_mid_5s', 'pca_factors'
            start_ts TIMESTAMPTZ NOT NULL,
            end_ts TIMESTAMPTZ NOT NULL,
            row_count INTEGER NOT NULL,
            file_size_bytes BIGINT,
            compression_type VARCHAR(20), -- 'lz4', 'snappy', 'gzip'
            version_id INTEGER REFERENCES market_data.curve_versions(version_id),
            checksum VARCHAR(64),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_accessed TIMESTAMPTZ,
            access_count INTEGER DEFAULT 0
        );
        
        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_manifest_data_type 
        ON market_data.archive_manifest (data_type, start_ts, end_ts);
        
        CREATE INDEX IF NOT EXISTS idx_manifest_s3_key 
        ON market_data.archive_manifest (s3_key);
    """)


def downgrade() -> None:
    """Drop curve-related tables."""
    op.execute("DROP TABLE IF EXISTS market_data.archive_manifest CASCADE;")
    op.execute("DROP TABLE IF EXISTS market_data.curve_versions CASCADE;")
    op.execute("DROP TABLE IF EXISTS market_data.pca_factors_archive CASCADE;")
    op.execute("DROP TABLE IF EXISTS market_data.curve_mid_5s_historical CASCADE;")