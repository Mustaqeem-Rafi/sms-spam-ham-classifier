"""Update all indexes (TOC, List of Figures, List of Tables) and fields of a .docx with
LibreOffice, then export it to PDF.

Uses a LibreOffice Basic macro installed into a throw-away LibreOffice profile, so it
needs only the `soffice` executable (no LibreOffice-bundled Python).

    python report/lo_export.py in.docx out.pdf
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MACRO = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE script:module PUBLIC "-//OpenOffice.org//DTD OfficeDocument 1.0//EN" "module.dtd">
<script:module xmlns:script="http://openoffice.org/2000/script" script:name="Module1" script:language="StarBasic">
Sub UpdateAndExport(sIn As String, sOut As String)
    Dim args(0) As New com.sun.star.beans.PropertyValue
    args(0).Name = "Hidden"
    args(0).Value = True
    oDoc = StarDesktop.loadComponentFromURL(ConvertToURL(sIn), "_blank", 0, args())
    For k = 1 To 3
        oIdx = oDoc.getDocumentIndexes()
        For i = 0 To oIdx.getCount() - 1
            oIdx.getByIndex(i).update()
        Next i
        oDoc.getTextFields().refresh()
        oDoc.refresh()
    Next k
    Dim pdf(0) As New com.sun.star.beans.PropertyValue
    pdf(0).Name = "FilterName"
    pdf(0).Value = "writer_pdf_Export"
    oDoc.storeToURL(ConvertToURL(sOut), pdf())
    oDoc.close(True)
    StarDesktop.terminate()
End Sub
</script:module>
"""


def find_soffice():
    for c in (os.environ.get("SOFFICE"), shutil.which("soffice"),
              "/Applications/LibreOffice.app/Contents/MacOS/soffice", "/usr/bin/soffice",
              "/usr/lib/libreoffice/program/soffice"):
        if c and os.path.exists(c):
            return c
    return None


def export(src, dst, timeout=600):
    soffice = find_soffice()
    if soffice is None:
        raise RuntimeError("LibreOffice (soffice) not found")
    src, dst = str(Path(src).resolve()), str(Path(dst).resolve())
    profile = Path(tempfile.mkdtemp(prefix="lo_profile_"))
    env_arg = f"-env:UserInstallation={profile.as_uri()}"
    try:
        # 1) initialise the profile, 2) drop the macro into My Macros/Standard, 3) run it
        subprocess.run([soffice, env_arg, "--headless", "--terminate_after_init"],
                       capture_output=True, timeout=timeout)
        std = profile / "user" / "basic" / "Standard"
        if not (std / "Module1.xba").exists():
            raise RuntimeError(f"LibreOffice profile was not initialised at {profile}")
        (std / "Module1.xba").write_text(MACRO, encoding="utf-8")
        if os.path.exists(dst):
            os.remove(dst)
        macro = f'macro:///Standard.Module1.UpdateAndExport("{src}","{dst}")'
        subprocess.run([soffice, env_arg, "--headless", "--invisible", "--nologo", "--norestore", macro],
                       capture_output=True, timeout=timeout)
        if not os.path.exists(dst):
            raise RuntimeError("LibreOffice did not produce the PDF")
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    return dst


if __name__ == "__main__":
    print("wrote", export(sys.argv[1], sys.argv[2]))
