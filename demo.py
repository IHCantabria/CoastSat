from ih import slopealt
import os


filepath_data = os.path.join(os.getcwd(), "data")
sitename = "NARRA"

narra = slopealt.beach_slope(filepath_data, sitename)