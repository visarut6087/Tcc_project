import openpyxl, glob, json
out = open("_probe_out.txt", "w", encoding="utf-8")
for p in sorted(glob.glob("memory_input/*.xlsx")):
    out.write("="*80+"\nFILE: "+p+"\n")
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    out.write("SHEETS: "+repr(wb.sheetnames)+"\n")
    for ws in wb.worksheets:
        out.write(f"-- sheet={ws.title!r} max_row={ws.max_row} max_col={ws.max_column}\n")
        n=0
        for row in ws.iter_rows(values_only=True):
            vals=[(str(v)[:45] if v is not None else "") for v in row]
            # trim trailing empties
            while vals and vals[-1]=="": vals.pop()
            out.write(f"  R{n}: {json.dumps(vals, ensure_ascii=False)}\n")
            n+=1
            if n>=10: break
    wb.close()
out.close()
print("DONE")
