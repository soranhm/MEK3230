import numpy as np
import sys




FREECADPATH = '/Applications/FreeCAD.app/Contents/lib/'
def import_fcstd(path_freecad):
    sys.path.append(path_freecad)
    try:
        import FreeCAD
    except:
        print "Kunne ikke importere FreeCAD"
        print "Bruker du riktig path?"

import_fcstd(FREECADPATH)

# navn av doc
freecad_doc_name = "test_scripting"
# hvor skal den lagres?
freecad_doc_path = "/Users/soranhussein/Dropbox/5_semseter/mek3230/"
# lagre som
freecad_extension = ".fcstd"

# liten eksempel
working_doc = FreeCAD.newDocument(freecad_doc_name)
# lager cylinger
cylinder_1 = working_doc.addObject("Part::Cylinder", 'cylinder_1')
# endre radius
cylinder_1.Radius = 4
# lagre documentet
working_doc.saveAs(freecad_doc_path + freecad_doc_name + freecad_extension)
