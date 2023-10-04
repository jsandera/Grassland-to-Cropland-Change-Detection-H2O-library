# Auto Machine Learning H2O Classification of Remote Sensing Data

library(raster)
library(rgdal)
library(caret)
library(h2o)
library(rChoiceDialogs)

# cpu core detection
h2o.init(nthreads=-1, max_mem_size="60G")
h2o.removeAll()

# set a working directory
vstupni_adresar <- rchoose.dir(getwd(), "Choose the Working Directory")
prac_adresar <- setwd(vstupni_adresar)

# choose raster file for classification
r <- brick(rchoose.files(caption = "Choose Raster Data for Classification"))

# prejmenovani kanalu rastru
names(r) <- c("August_30_2015", "August_4_2016", "June_20_2017", "September_28_2017", "April_6_2018",
              "April_21_2018", "May_31_2018", "August_29_2018", "September_18_2018", "September_28_2018",
              "October_13_2018", "April_1_2019", "April_16_2019", "June_30_2019", "August_29_2019", "April_5_2020",
              "April_20_2020", "July_14_2020", "July_24_2020", "August_8_2020", "August_13_2020", "August_28_2020",
              "September_12_2020")

# select directory, where taining points are located
vyber_shp <- rchoose.dir(getwd(), "Choose Directory of Training Points' Location")

# start time of computations
start <- Sys.time()

# transformation training data into data frame
shp <- readOGR(dsn = vyber_shp, layer = "training_points")

# traning data separation - 50% for traning, 50% for validation
dataSeparation <- createDataPartition(shp$Class, p = .5)[[1]]
shpTrain <- shp[dataSeparation,]
shpTest <- shp[-dataSeparation,]

# traning data extraction and transformation into data frame
shpTrain1 <- na.omit(as.data.frame(cbind(shpTrain, extract(r, shpTrain))))
sloupecky <- ncol(shpTrain1)-2
shpTrain2 <- shpTrain1[,2:sloupecky]
shpTrain2[,1] <- as.factor(shpTrain2[,1])

# traning data split for crossvalidation - 70% traning, 30% validation (Training, Validation, Test)
df <- as.h2o(shpTrain2)
raster_b <- rasterToPoints(r, progress="text")
souradnice <- as.data.frame(raster_b[,1:2])
data <- as.data.frame(raster_b[,3:ncol(raster_b)])
grid <- as.h2o(data)

splits <- h2o.splitFrame(df, ratios=0.70, seed=1234)
train <- h2o.assign(splits[[1]], "train.hex") # 70%
valid  <- h2o.assign(splits[[2]], "valid.hex") # 30%


# define a response column in training data frame
response <- "Class"
predictors <- setdiff(names(train), response)
predictors

# model construction with AutoMl function
m <- h2o.automl(x=predictors, y=response, training_frame=train, validation_frame=valid, nfolds=10, max_runtime_secs=3600,
                max_models=20)

# Save Model Performance Results into hdd in text file format
pt <- capture.output(m)
ul <- cat(pt, file="Model_Protocol.txt", sep="\n", append=FALSE)

# Predict properly tuned machine learning model (the best one in leader board)
pr <- as.data.frame(h2o.predict(m, grid))
pr

# Generate Variable Importance Heatmap
var_imp_heatmap <- h2o.varimp_heatmap(m)
print(var_imp_heatmap)

# Export Heatmap in hdd to working directory
jpeg(filename="Variable_Importance_AUTOML_HEATMAP_H2O.jpeg", units="px", width=3000, height=3000, res=600)
print(var_imp_heatmap)
dev.off()

# Crerate Thematic Raster from Output Prediction of AutoML Model
predicted_raster <- cbind(souradnice, pr$predict)
R <- rasterFromXYZ(predicted_raster)
plot(R)

# Classification Export to disk - Erdas Imagine File Format
exp <-  writeRaster(R, filename="KlasifikaceAUTOML_H2O.img", format='HFA', overwrite = TRUE)

###############################################################################################################
###############################################################################################################

# Accuracy Assessment

# Test data extraction
shpTest1 <- as.data.frame(cbind(shpTest, extract(exp, shpTest)))

# Confusion Matrix Construction
pred <- as.factor(shpTest1[,3])
val  <- as.factor(shpTest$Class)
hodnoty <- data.frame(pred, val)

# Confusin Matrix Generation
cm1 <- confusionMatrix(data=hodnoty$pred, 
                       reference=hodnoty$val)
cm1

# Export Confusion Matrix to hdd
zaznam1 <- capture.output(cm1)
presnost1 <- cat(zaznam1, file="Chybova_Matice_AUTOML_H2O.txt", sep="\n", append=FALSE)

# Confusion Matrix Data Type Change
M <- as.matrix(cm1)

# Producer's Accuracy
Z <- diag(M)/colSums(M)
Z

# Users Accuracy
U <- diag(M)/rowSums(M)
U

# Producer's Accuracy Export
zaznam2 <- capture.output(Z)
presnost2 <- cat(zaznam2, file="PRODUCERS_ACCURACY_AUTOML.txt", sep="\n", append=FALSE)

# User's Accuracy Export
zaznam3 <- capture.output(U)
presnost3 <- cat(zaznam3, file="USERS_ACCURACY_AUTOML.txt", sep="\n", append=FALSE)

# ROC Curve Construction
model_is <- as.vector(m@leaderboard$model_id)
index <- 1
model_e <- h2o.getModel(model_is[index])
roc <- plot(h2o.performance(model_e, valid), type="roc")

jpeg(filename="ROC_Curve.jpeg", units="px", width=3000, height=3000, res=600)
roc2 <- plot(h2o.performance(model_e, valid), type="roc")
dev.off()

# save training and validation point datasets into hard drive in shapefile format
tr <-  writeOGR(obj=shpTrain, dsn=getwd(), layer="Training Points.shp", driver="ESRI Shapefile")
val <- writeOGR(obj=shpTest, dsn=getwd(), layer="Validation Points.shp", driver="ESRI Shapefile")

#########################################################################################################################################################

# save all tested models in H2O automl

# Create dedicated working directory

output_dir <- file.path(getwd(), "Trained Models")

if (!dir.exists(output_dir)){
  dir.create(output_dir)
} else {
  print("Dir already exists!")
}

# create folder for all trained model export
new_wd <- setwd(output_dir)

# create folder for all trained model export
mod_ids <- m@leaderboard$model_id

# export all trained models to disk
for(i in 1:nrow(mod_ids)) {
  
  aml1 <- h2o.getModel(m@leaderboard[i, 1]) # get model object in environment
  h2o.save_mojo(object = aml1, path=getwd(), force=TRUE) # pass that model object to h2o.saveModel as an argument
  
}

# final computation time
konec <- Sys.time()
cas <- konec - start
cas

# Export Time Elapsed for Computation
z_cas <- capture.output(cas)
u_cas <- cat(z_cas, file="Time_Ellapsed_AUTOML.txt", sep="\n", append=FALSE)







