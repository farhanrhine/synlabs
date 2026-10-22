from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Groq API
    groq_api_key: str = ""
    
    # Judge Model
    judge_model: str = "openai/gpt-oss-120b"
    judge_temperature: float = 0.0

    # Generator Models for A/B Testing
    generator_model_a: str = "openai/gpt-oss-120b"
    generator_model_b: str = "qwen/qwen3.8-27b"

    # Evaluation Thresholds
    default_pass_threshold: float = 3.5
    position_swap_enabled: bool = True

    # Storage Paths
    artifacts_dir: str = "./artifacts"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
