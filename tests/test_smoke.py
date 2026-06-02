def test_import_and_cli_main_exists():
    from selecta import cli

    assert hasattr(cli, "main")


def test_cli_help_runs():
    from selecta import cli

    # no command -> prints help, returns 0
    assert cli.main([]) == 0
