def test_robot_engine_from_config():
    """RobotEngine can be constructed from a config dict."""
    from glados.robot.config import RobotConfig
    from glados.robot.engine import RobotEngine

    config = RobotConfig(
        llm_model="gemma3:4b",
        completion_url="http://localhost:11434/api/chat",
        asr_engine="whisper",
        voice="glados_ru",
        personality="Test.",
    )
    # Engine construction should not crash (won't start threads)
    engine = RobotEngine(config, start_vision=False, start_audio=False)
    assert engine._config.llm_model == "gemma3:4b"
    assert engine._shutdown is not None
