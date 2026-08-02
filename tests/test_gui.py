def test_gui_module_import_does_not_require_ui_extras() -> None:
    import rotarycam.gui as gui

    assert callable(gui.main)
    assert callable(gui.create_main_window)
