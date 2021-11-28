In addition to the model data, you will find a metadata.csv (comma-separated value format) file that contains the metadata associated with each model.  The columns of this file and their interpretation are as follows:

fullId : the unique id of the model
category : manually annotated categories to which this model belongs
wnsynset : comma-separated list of WordNet synset offsets to which the model belongs
wnlemmas : comma-separated list of names (lemmas) of the WordNet synsets to which the model belongs
up : normalized vector in original model space coordinates indicating semantic "upright" orientation of model
front : normalized vector in original model space coordinates indicating semantic "front" orientation of model
unit : scale unit converting model virtual units to meters
aligned.dims : aligned dimensions of model after rescaling to meters and upright-front realignment (X-right, Y-back, Z-up)
isContainerLike : whether this model is container-like (i.e., is internally empty)
surfaceVolume : total volume of surface voxelization of mesh (m^3) used for container-like objects
solidVolume : total volume of solid (filled-in) voxelization of mesh (m^3) used for non container-like objects
supportSurfaceArea : surface area of support surface (usually bottom) (m^2)
weight : estimated weight of object in Kg computed from material priors and appropriate volume
staticFrictionForce : static friction force required to push object computed from supportSurfaceArea and coefficient of static friction using material priors
name : name of the model as indicated on original model repository (uncurated)
tags : tags assigned to the model on original model repository (uncurated)

In addition to the existing CSV file, you can re-download the current metadata from the ShapeNet server through the following URL:
https://www.shapenet.org/solr/models3d/select?q=isAligned%3Atrue+AND+source%3Awss+AND+category%3A*&rows=100000&fl=fullId%2Ccategory%2Cwnsynset%2Cwnlemmas%2Cup%2Cfront%2Cunit%2Caligned.dims%2CisContainerLike%2CsurfaceVolume%2CsolidVolume%2CsupportSurfaceArea%2Cweight%2CstaticFrictionForce%2Cname%2Ctags&wt=csv&indent=true

Please note that the above link restricts the retrieved metadata to only models that have manually verified alignments and categorizations (currently about half of the total). If you would like the full list of all models, use the following link: 
https://www.shapenet.org/solr/models3d/select?q=source%3Awss&rows=100000&fl=fullId%2Ccategory%2Cwnsynset%2Cwnlemmas%2Cup%2Cfront%2Cunit%2Caligned.dims%2CisContainerLike%2CsurfaceVolume%2CsolidVolume%2CsupportSurfaceArea%2Cweight%2CstaticFrictionForce%2Cname%2Ctags&wt=csv&indent=true
