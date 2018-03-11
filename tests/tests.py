# Here are some test functions not conforming to any software engineering conventions

# I would need to write instrument emulators in order to write actual tests?

import sys
import os
ivtoolsdir = os.path.join(os.path.split(sys.path[0])[0], 'ivtools')

def name_collisions():
    # Interactive script imports all names into the same namespace
    # So make sure there are no name collisions
    import ast

    filenames = ['measure.py', 'io.py', 'plot.py', 'analyze.py']
    names = []
    for fn in filenames:
        with open(os.path.join(ivtoolsdir, fn)) as file:
            node = ast.parse(file.read())

        functions = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        funcnames = [f.name for f in functions]
        classes = [n for n in node.body if isinstance(n, ast.ClassDef)]
        classnames = [c.name for c in classes]
        names.append(set(funcnames + classnames))

    fail = False
    from itertools import combinations
    for (i, j) in combinations(range(4), 2):
        intersect = names[i].intersection(names[j])
        if any(intersect):
            print('Name collision!')
            print(filenames[i])
            print(filenames[j])
            print(intersect)
            fail = True

    return fail

if __name__ == '__main__':
    name_collisions()
