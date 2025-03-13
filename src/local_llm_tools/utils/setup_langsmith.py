import json
import os


def setup_langsminth_credentials(project_name: str):
    """
    下みたいなjsonファイル
    {
        "LANGSMITH_TRACING": true,
        "LANGSMITH_ENDPOINT": "https://api.smith.langchain.com",
        "LANGSMITH_API_KEY": <api_key>,
        "LANGSMITH_PROJECT": <project_id>
    }
    """
    file_path = os.path.join(os.environ["HOME"], "langsmith", "config.json")
    with open(file_path) as f:
        configs = json.load(f)[project_name]

    # 環境変数に設定
    for key, value in configs.items():
        os.environ[key] = value
