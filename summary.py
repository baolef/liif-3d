import os
import openpyxl


def add(model, epoch, test, metric, value):
    if metric not in results.keys():
        results[metric] = {}
    if model not in results[metric].keys():
        results[metric][model] = {}
    if epoch not in results[metric][model].keys():
        results[metric][model][epoch] = {}
    results[metric][model][epoch][test] = value


if __name__ == '__main__':
    root = 'results'
    results = {}
    for model in os.listdir(root):
        path = os.path.join(root, model)
        if os.path.isfile(path):
            continue
        if model.startswith('_'):
            for epoch in os.listdir(path):
                path = os.path.join(root, model, epoch)
                for test in os.listdir(path):
                    path = os.path.join(root, model, epoch, test, 'result.txt')
                    if os.path.exists(path):
                        with open(path) as f:
                            for line in f.readlines():
                                metric, value = line.split(':')
                                add(model, epoch, test, metric, float(value))
        else:
            epoch='N/A'
            for test in os.listdir(path):
                path = os.path.join(root, model, test, 'result.txt')
                if os.path.exists(path):
                    with open(path) as f:
                        for line in f.readlines():
                            metric, value = line.split(':')
                            add(model, epoch, test, metric, float(value))

    summary = {}
    for metric in results.keys():
        headers = ['model', 'epoch']
        tests = set()
        for model in results[metric].keys():
            for epoch in results[metric][model].keys():
                for test in results[metric][model][epoch].keys():
                    tests.add(test)
        headers = headers + sorted(list(tests), key=lambda x: int(x.split('-')[-1]))
        table = [headers]

        for model in results[metric].keys():
            for epoch in results[metric][model].keys():
                row = []
                row.append(model)
                row.append(epoch)
                for test in headers[2:]:
                    if test in results[metric][model][epoch].keys():
                        row.append(results[metric][model][epoch][test])
                    else:
                        row.append('N/A')
                table.append(row)
        summary[metric] = table

    wb = openpyxl.Workbook()
    for metric, table in summary.items():
        ws = wb.create_sheet(metric)
        ws.title = metric
        n_rows = len(table)
        n_cols = len(table[0])
        for row in table:
            ws.append(row)
    wb.save(os.path.join(root, 'summary.xlsx'))
