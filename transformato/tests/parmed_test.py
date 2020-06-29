from parmed.charmm import CharmmParameterSet

print ('std')
params_std = CharmmParameterSet('/home/master/debug/std.prm', '/home/master/debug/std.rtf')
print (params_std.bond_types[('CG331' , 'CG331')])
print (params_std.bond_types[('CG331' , 'HGA3')])

print (params_std.angle_types[('CG331' , 'CG331' , 'HGA3')])
print (params_std.angle_types[('HGA3' , 'CG331' , 'HGA3')])

print (params_std.dihedral_types[('HGA3' , 'CG331' , 'CG331' ,'HGA3')])


print ('non std')
params_non_std = CharmmParameterSet('/home/master/debug/non_std.prm', '/home/master/debug/non_std.rtf')
print (params_non_std.bond_types[('CG331' , 'CG3311')])
print (params_non_std.bond_types[('CG331' , 'HGA3')])
print (params_non_std.bond_types[('CG3311' , 'HGA3')])

print (params_non_std.angle_types[('CG3311' , 'CG331' , 'HGA3')])
print (params_non_std.angle_types[('CG331' , 'CG3311' , 'HGA3')])
print (params_non_std.angle_types[('HGA3' , 'CG331' , 'HGA3')])
print (params_non_std.angle_types[('HGA3' , 'CG3311' , 'HGA3')])

print (params_non_std.dihedral_types[('HGA3' , 'CG331' , 'CG3311' ,'HGA3')])
print (params_non_std.dihedral_types[('HGA3' , 'CG3311' , 'CG331' ,'HGA3')])

print ('non std2')
params_non_std2 = CharmmParameterSet('/home/master/debug/non_std2.prm', '/home/master/debug/non_std2.rtf')
print (params_non_std2.bond_types[('CG3312' , 'CG3311')])
print (params_non_std2.bond_types[('CG3312' , 'HGA3')])
print (params_non_std2.bond_types[('CG3311' , 'HGA3')])

print (params_non_std2.angle_types[('CG3311' , 'CG3312' , 'HGA3')])
print (params_non_std2.angle_types[('CG3312' , 'CG3311' , 'HGA3')])
print (params_non_std2.angle_types[('HGA3' , 'CG3312' , 'HGA3')])
print (params_non_std2.angle_types[('HGA3' , 'CG3311' , 'HGA3')])

print (params_non_std2.dihedral_types[('HGA3' , 'CG3312' , 'CG3311' ,'HGA3')])
print (params_non_std2.dihedral_types[('HGA3' , 'CG3311' , 'CG3312' ,'HGA3')])

assert(id(params_non_std.dihedral_types[('HGA3' , 'CG331' , 'CG3311' ,'HGA3')]) == id(params_non_std.dihedral_types[('HGA3' , 'CG3311' , 'CG331' ,'HGA3')])) == True
assert(id(params_non_std2.dihedral_types[('HGA3' , 'CG3312' , 'CG3311' ,'HGA3')]) == id(params_non_std2.dihedral_types[('HGA3' , 'CG3311' , 'CG3312' ,'HGA3')])) == True