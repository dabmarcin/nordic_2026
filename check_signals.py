from nordic_config import PORTFOLIO_SIGNALS

print('=== AKTUALNY PORTFOLIO_SIGNALS ===')
for k, v in sorted(PORTFOLIO_SIGNALS.items()):
    print(f'{k:<25} {v.get("label", "?"):<35} [{v["tier"]}]')
