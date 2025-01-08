mlx_lm.server \
--model /Users/Zhiyuan/Project/rag_projects/my-web-novel/models/Qwen2.5-7B-Instruct-8b \
--host 127.0.0.1 --port 8889 \
--log-level INFO \
--use-default-chat-template


# add `seed` in the `server.py``from mlx_lm repo
#
# usage: mlx_lm.server [-h] [--model MODEL] [--adapter-path ADAPTER_PATH] [--host HOST] [--port PORT] [--trust-remote-code] [--log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
#                      [--cache-limit-gb CACHE_LIMIT_GB] [--chat-template CHAT_TEMPLATE] [--use-default-chat-template]

# MLX Http Server.

# optional arguments:
#   -h, --help            show this help message and exit
#   --model MODEL         The path to the MLX model weights, tokenizer, and config
#   --adapter-path ADAPTER_PATH
#                         Optional path for the trained adapter weights and config.
#   --host HOST           Host for the HTTP server (default: 127.0.0.1)
#   --port PORT           Port for the HTTP server (default: 8080)
#   --trust-remote-code   Enable trusting remote code for tokenizer
#   --log-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
#                         Set the logging level (default: INFO)
#   --cache-limit-gb CACHE_LIMIT_GB
#                         Set the MLX cache limit in GB
#   --chat-template CHAT_TEMPLATE
#                         Specify a chat template for the tokenizer
#   --use-default-chat-template
#                         Use the default chat template