"""Static branch binding regression; real lifecycle coverage is a separate CLI gate."""
import ast
from pathlib import Path
import symtable
import unittest

ROOT=Path(__file__).resolve().parents[1]


class EntryBindings(unittest.TestCase):
    def test_main_does_not_shadow_module_functions(self):
        raw=(ROOT/'softnms_job.py').read_text(encoding='utf-8')
        tree=ast.parse(raw)
        main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
        names={x.name for x in tree.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
        bindings=[]
        for node in ast.walk(main):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                bindings += [a.asname or a.name.split('.')[0] for a in node.names]
        self.assertFalse(names.intersection(bindings))
        table=next(x for x in symtable.symtable(raw,'softnms_job.py','exec').get_children() if x.get_name()=='main')
        self.assertTrue(table.lookup('supervise').is_global())

    def test_launcher_import_has_no_execution(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('softnms_launch_import_check',ROOT/'softnms_score_launch.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertTrue(callable(module.main))


if __name__=='__main__':unittest.main()
