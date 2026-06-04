import sys
from pathlib import Path

# Adiciona CoreRefactoryLangGraph/ ao path para imports diretos (chains, nodes, state, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
