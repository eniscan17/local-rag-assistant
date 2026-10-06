"""
Chat-model backends for the answer-quality benchmark.

    foundry:<alias>   the app's own Foundry Local path (Apple Silicon Mac),
                      e.g. foundry:phi-3.5-mini, foundry:qwen2.5-0.5b
    hf:<model id>     Hugging Face transformers on CPU/GPU, for machines
                      without Foundry Local, e.g. hf:Qwen/Qwen2.5-0.5B-Instruct

All backends decode greedily (temperature 0 where the backend allows it),
so a re-run reproduces the same answers.
"""


class Generator:
    name: str

    def generate(self, system: str, user: str) -> str: ...


class FoundryGenerator(Generator):
    def __init__(self, alias: str, max_tokens: int):
        from foundry_local_sdk import Configuration, FoundryLocalManager

        import config

        try:
            FoundryLocalManager.initialize(Configuration(app_name=config.APP_NAME))
        except Exception:
            pass  # already initialised in this process
        manager = FoundryLocalManager.instance
        self.model = manager.catalog.get_model(alias)
        self.model.download(lambda p: print(f"\rDownloading {alias}: {p:.1f}%", end="", flush=True))
        self.model.load()
        self.client = self.model.get_chat_client()
        self.client.settings.max_tokens = max_tokens
        try:
            self.client.settings.temperature = 0.0
        except Exception:
            print(f"note: could not set temperature=0 for {alias}; answers may vary between runs")
        self.name = f"foundry:{alias}"

    def generate(self, system, user):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        parts = []
        for chunk in self.client.complete_streaming_chat(messages):
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
        return "".join(parts).strip()


class HFGenerator(Generator):
    def __init__(self, model_id: str, max_tokens: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
        self.model.eval()
        self.max_tokens = max_tokens
        self.name = f"hf:{model_id}"

    def generate(self, system, user):
        import torch

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        inputs = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                              return_tensors="pt", return_dict=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def make_generator(spec: str, max_tokens: int) -> Generator:
    backend, _, name = spec.partition(":")
    if backend == "foundry":
        return FoundryGenerator(name, max_tokens)
    if backend == "hf":
        return HFGenerator(name, max_tokens)
    raise ValueError(f"unknown generator {spec!r}; use foundry:<alias> or hf:<model id>")
