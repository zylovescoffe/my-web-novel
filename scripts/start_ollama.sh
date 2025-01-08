# OLLAMA_KEEP_ALIVE=-1 /Users/Zhiyuan/Project/ollama/ollama serve
# OLLAMA_FLASH_ATTENTION=1 OLLAMA_KEEP_ALIVE=-1 /Users/Zhiyuan/Project/ollama/ollama serve
# OLLAMA_KEEP_ALIVE=-1 OLLAMA_KV_CACHE_TYPE=q8_0 /Users/Zhiyuan/Project/ollama/ollama serve
# OLLAMA_FLASH_ATTENTION=1 OLLAMA_KEEP_ALIVE=-1 OLLAMA_KV_CACHE_TYPE=q8_0 /Users/Zhiyuan/Project/ollama/ollama serve


# Experiement Result with Ollama
# noFA-f16kv-f16model: 13.02s@65t , 11.9s@41t
# noFA-f16kv-q8model:  13.45s@65t , 12.4s@39t
# FA-f16kv-f16model:   12.15s@65t , 11.4s@41t
# FA-f16kv-q8model:    12.57s@65t , 11.9s@41t
# Quantized kv not able to work in noFA case
# FA-q8kv-f16model:    17.07s@63t , 15.9s@34t
# FA-q8kv-q8model:     17.42s@65t , 16.5s@34t


OLLAMA_FLASH_ATTENTION=1 OLLAMA_KEEP_ALIVE=-1 /Users/Zhiyuan/Project/ollama/ollama serve