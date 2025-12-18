"""
Apache Sedona and Spark session setup.
"""

import os
from typing import Optional
from pyspark.sql import SparkSession


def get_spark_session(
    app_name: str = "geosuite-geospatial",
    local_mode: bool = True,
    memory: str = "4g",
    cores: str = "*"
) -> SparkSession:
    """
    Create or get a SparkSession configured for Sedona.
    
    Args:
        app_name: Application name
        local_mode: If True, run in local mode
        memory: Driver and executor memory (e.g., "4g")
        cores: Number of cores for local mode (e.g., "*" for all)
        
    Returns:
        Configured SparkSession
    """
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryo.registrator", "org.apache.sedona.core.serde.SedonaKryoRegistrator")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.memory", memory)
    )
    
    # Local mode configuration
    if local_mode:
        builder = builder.master(f"local[{cores}]")
    
    # Sedona-specific configs for spatial joins
    builder = (
        builder
        .config("sedona.join.gridtype", "KDBTREE")
        .config("sedona.join.indexbuild", "true")
        .config("sedona.join.numpartition", "8")
    )
    
    spark = builder.getOrCreate()
    
    # Set log level to WARN to reduce verbosity
    spark.sparkContext.setLogLevel("WARN")
    
    return spark


def init_sedona(spark: SparkSession) -> None:
    """
    Initialize Sedona extensions on an existing SparkSession.
    
    Args:
        spark: SparkSession to initialize
    """
    from sedona.register import SedonaRegistrator
    
    SedonaRegistrator.registerAll(spark)


def get_or_create_session(
    app_name: str = "geosuite-geospatial",
    **kwargs
) -> SparkSession:
    """
    Get existing Sedona-enabled SparkSession or create a new one.
    
    This is the recommended entry point for getting a Spark session.
    
    Args:
        app_name: Application name
        **kwargs: Additional arguments passed to get_spark_session
        
    Returns:
        Configured SparkSession with Sedona registered
    """
    spark = get_spark_session(app_name=app_name, **kwargs)
    init_sedona(spark)
    return spark


def stop_spark_session(spark: Optional[SparkSession] = None) -> None:
    """
    Stop a SparkSession.
    
    Args:
        spark: SparkSession to stop, or None to stop the active session
    """
    if spark is None:
        spark = SparkSession.getActiveSession()
    
    if spark is not None:
        spark.stop()

