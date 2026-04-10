"""Teste de integração, verifica que o pipeline Kedro roda 100%"""

from pathlib import Path

from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


class TestKedroRun:
    def test_kedro_run(self):
        bootstrap_project(Path.cwd())

        with KedroSession.create(project_path=Path.cwd(), env="ci") as session:
            assert session.run() is not None
