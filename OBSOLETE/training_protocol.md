1. load dataset - each dataset has inputs (1200 geometries, 325 wavevectors in embedded form, 6 bands in embedded form) and outputs (1200x325x6 sets of displacement fields). This results in a 1200x325x6 dataset with each sample being a pairing of (3x32x32) input with (4x32x32) output. Let P = 32x32 and S = 7P = 7x32x32
   1.1 Known: RAM can handle 1800x325x6xS = 3510000xS = 3510000 x (3P + 4P).
   1.2 There are 40 datasets. Downselect 1/5 of the wavevectors and 1/2 of bands -> 40x1200x65x3xS = 9360000 x (3P + 4P) (about 3x over RAM)
   1.3 Load only one copy of the geometries, wavevectors, and bands -> 9360000 x 4P + ((40x1200) + 325 + 6) x P = 9372331 x 4P ~= 5355618 x 7P (about 1.5x over RAM)
   1.4 Use Pytorch to recast and save data as float8\_e4m3 to conserve half the bits, this dataset can in theory fit in RAM.



