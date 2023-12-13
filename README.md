## ASAPP : Annotation of Single-cell data by Approximate Pseudo-bulk projection


<div align="center">
    <img src="images/asapp_workflow.png" alt="Logo" width="500" height="600">
</div>


## Tutorial

A detail tutorial on how to run ASAPP can be found [here](https://github.com/causalpathlab/asapp/blob/main/asappy_tutorial.md).

ASAPP framework consists of the following steps-

```
# Step 1. create asap data
asappy.create_asap_data()

# Step 2. create asap object
asap_object = asappy.create_asap_object()

# Step 3. generate pseudo-bulk
asappy.generate_pseudobulk()

# Step 4. run NMF
asappy.asap_nmf()

# Step 5. save model
asappy.generate_model()

```