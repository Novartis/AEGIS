from mhciipresentation.transformer import prepare_iedb_data
train_data, val_data, test_data, y_train, y_val, y_test = prepare_iedb_data()

def ds_stats(ds, ys):

    num_EL_alleles_human=len(set([el[1]["Alleles"] for el in list(ds[ys!=0][ds[ys!=0]["file_name"].str.contains("_EL")].iterrows()) 
                                  if  not "mouse" in el[1]["MHC_molecule"]] ))
    num_EL_alleles_mouse=len(set([el[1]["Alleles"] for el in list(ds[ys!=0][ds[ys!=0]["file_name"].str.contains("_EL")].iterrows())
                                  if "mouse" in el[1]["MHC_molecule"]] ))

    num_EL_exp_peptides_human = len([el for el in list(ds[ys!=0][ds[ys!=0]["file_name"].str.contains("_EL")]["MHC_molecule"]) if not "mouse" in el] )
    num_EL_exp_peptides_mouse = len([el for el in list(ds[ys!=0][ds[ys!=0]["file_name"].str.contains("_EL")]["MHC_molecule"]) if "mouse" in el] )

    num_EL_syn_peptides_human = len([el for el in list(ds[ys==0][ds[ys==0]["file_name"].str.contains("_EL")]["MHC_molecule"]) if not "mouse" in el] )
    num_EL_syn_peptides_mouse = len([el for el in list(ds[ys==0][ds[ys==0]["file_name"].str.contains("_EL")]["MHC_molecule"]) if "mouse" in el] )

    num_BA_exp_peptides_human = len([el for el in list(ds[ds["file_name"].str.contains("_BA")]["MHC_molecule"]) if not "H-2" in el] )
    num_BA_exp_peptides_mouse =len([el for el in list(ds[ds["file_name"].str.contains("_BA")]["MHC_molecule"]) if "H-2" in el] )
    
    num_BA_exp_alleles_human = len(set([el[1]["Alleles"] for el in list(ds[ys!=0][ds[ys!=0]["file_name"].str.contains("_BA")].iterrows()) 
                                        if  not "H-2" in el[1]["MHC_molecule"]] ))
    num_BA_exp_alleles_mouse = len(set([el[1]["Alleles"] for el in list(ds[ys!=0][ds[ys!=0]["file_name"].str.contains("_BA")].iterrows()) 
                                        if "H-2" in el[1]["MHC_molecule"]] ))

    print("IEDB(H) EXP BA(n=%s): %s" %(num_BA_exp_alleles_human,num_BA_exp_peptides_human))
    print("IEDB(H) EXP EL(n=%s): %s" %(num_EL_alleles_human,num_EL_exp_peptides_human))
    print("IEDB(H) SYN EL      : %s" %num_EL_syn_peptides_human)

    print("IEDB(M) EXP BA(n=%s): %s" %(num_BA_exp_alleles_mouse,num_BA_exp_peptides_mouse))
    print("IEDB(M) EXP EL(n=%s): %s" %(num_EL_alleles_mouse,num_EL_exp_peptides_mouse))
    print("IEDB(M) SYN EL      : %s" %num_EL_syn_peptides_mouse)

print("TRAIN")
ds_stats(train_data,y_train)
print("VAL")
ds_stats(val_data,y_val)
print("TEST")
ds_stats(test_data,y_test)

