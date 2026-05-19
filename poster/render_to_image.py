"""Render PosterA045R_filled.pptx to PDF then JPG via PowerPoint COM."""
import os, sys
import win32com.client
import pythoncom

HERE = os.path.dirname(os.path.abspath(__file__))
PPTX = os.path.join(HERE, "PosterA045R_filled.pptx")
PDF  = os.path.join(HERE, "PosterA045R_filled.pdf")
PNG  = os.path.join(HERE, "PosterA045R_filled.png")

pythoncom.CoInitialize()
ppt = win32com.client.Dispatch("PowerPoint.Application")
# Some versions don't allow setting Visible=False
try:
    ppt.Visible = False
except Exception:
    pass

pres = ppt.Presentations.Open(PPTX, WithWindow=False)
try:
    # Export to PDF
    pres.SaveAs(PDF, 32)  # 32 = ppSaveAsPDF
    print("PDF:", PDF)
    # Export slide 1 to PNG
    pres.Slides[0].Export(PNG, "PNG", 2400, 1680)
    print("PNG:", PNG)
finally:
    pres.Close()
    ppt.Quit()
