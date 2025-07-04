pacman::p_load('dplyr', 'plyr','stringr', 'mvtnorm','MASS','data.table', 'sampling')

generateGigantiData <- function(beta.X = -1, beta.Y = 0.25,
                                gamma.X = -2, gamma.Y = 0.5,
                                covXY = 0.25, digit = "0001"){
  n <- 4000
  beta.unval<-c(-7,-0.02,0,0,6.5)
  gamma.unval<-c(-8,-0.02,-0,0,5,0.5)
  beta<-c(-5,-0.02,beta.X,beta.Y,0,5,0.5,0)
  gamma<-c(-6.5,-0.02,gamma.X,gamma.Y,0,0.0,5.0,0,0.5)
  Intercept<-rep(1,n)
  XYcov<-covXY
  Xvar<-1
  Yvar<-1
  maxtime<-100
  time<-maxtime
  times<-round((0:maxtime)*30.437/365.25,3)
  ID<-rep(1:n,time)
  cubs.unval<-(-2)
  LP.visit.unval<-cubs.unval
  p.visit.unval<-exp(LP.visit.unval)/(1+ exp(LP.visit.unval))
  C.star<-rbinom(n*maxtime,1,p.visit.unval)
  
  cubs<-(-5 + 9*C.star)
  LP.visit<-cubs
  p.visit<-exp(LP.visit)/(1+ exp(LP.visit))
  C<-rbinom(n*maxtime,1,p.visit)
  
  
  #Covariates X and Y
  XY<-rmvnorm(n, mean=c(0,0),sigma=matrix(c(Xvar,XYcov,XYcov,Yvar),2,2))
  X<-XY[,1]
  Y<-XY[,2]
  Xmat<-cbind(1,sort(rep(1:maxtime,n)),X,Y)
  rownames(Xmat)<-ID
  colnames(Xmat)<-c("Intercept","time","X","Y")
  
  ####################################
  #Unvalidated data
  ####################################
  #ART(unvalidated)	
  LP.ART.unval<-Xmat%*%beta.unval[1:4] + C.star*beta.unval[5]
  p.ART.unval<- exp(LP.ART.unval)/(1+ exp(LP.ART.unval))
  
  #Indicator of ART at time t
  e.ART.unval<-rbinom(n*maxtime,1,p.ART.unval)
  A.star<-matrix(e.ART.unval,n,maxtime,byrow=FALSE)
  
  
  #ADE(unvalidated)		
  LP.ADE.unval<-Xmat%*%gamma.unval[1:4]  + C.star*gamma.unval[5] + e.ART.unval*gamma.unval[6]
  p.ADE.unval<-exp(LP.ADE.unval)/(1+ exp(LP.ADE.unval))
  #Indicator of event (ADE) at time t
  e.ADE.unval<-rbinom(n*maxtime,1,p.ADE.unval)
  D.star<-matrix(e.ADE.unval,n,maxtime,byrow=FALSE)
  
  ####################################
  ######Validated data
  ####################################
  
  #ART(validated)
  LP.ART<-Xmat%*%beta[1:4] + C.star*beta[5] + e.ART.unval*beta[6] + e.ADE.unval*beta[7] + C*beta[8]
  p.ART<- exp(LP.ART)/(1+ exp(LP.ART))
  #Indicator of ART at time t
  e.ART<-rbinom(n*maxtime,1,p.ART)
  A<-matrix(e.ART,n,maxtime,byrow=FALSE)
  
  
  
  #ADE(validated)
  LP.ADE<-Xmat%*%gamma[1:4] + C.star*gamma[5] + e.ART.unval*gamma[6] + e.ADE.unval*gamma[7]+ C*gamma[8] + e.ART*gamma[9]
  p.ADE<- exp(LP.ADE)/(1+ exp(LP.ADE))
  #Indicator of event (ADE) at time t
  e.ADE<-rbinom(n*maxtime,1,p.ADE)
  D<-matrix(e.ADE,n,maxtime,byrow=FALSE)
  
  truth.long<-data.frame(ID,C,cbind(sort(rep(1:maxtime,n)),Intercept,X,Y,as.vector(A),as.vector(D)))
  unvalidated.long<-data.frame(ID,C.star,cbind(sort(rep(1:maxtime,n)),Intercept,X,Y,as.vector(A.star),as.vector(D.star)))
  colnames(truth.long)<-c("ID","C","time","Intercept","X","Y","A","D")
  colnames(unvalidated.long)<-c("ID","C.star","time","Intercept","X","Y","A.star","D.star")
  
  censor<-ddply(truth.long[truth.long$C==1 | truth.long$time==1,],.(ID),summarize,lastC=max(time))
  censor.star<-ddply(unvalidated.long[unvalidated.long$C.star==1 | unvalidated.long$time==1,],.(ID),summarize,lastC.star=max(time))
  #truth.long<-merge(truth.long,censor,by="ID")
  #unvalidated.long<-merge(unvalidated.long,censor.star,by="ID")
  aa<-data.table(data.frame(truth.long),key=c("ID"))
  bb<-data.table(data.frame(censor),key=c("ID"))
  truth.long<-data.frame(aa[bb,])
  cc<-data.table(data.frame(unvalidated.long),key=c("ID"))
  dd<-data.table(data.frame(censor.star),key=c("ID"))
  unvalidated.long<-data.frame(cc[dd,])
  
  dat.original<-data.frame(unvalidated.long,truth.long[,c("C","lastC","A","D")])
  dat.original<-dat.original[order(dat.original$ID,dat.original$time),]
  
  dat.original$C.star[dat.original$lastC.star<dat.original$time]<-0
  dat.original$A.star[dat.original$lastC.star<dat.original$time]<-0
  dat.original$D.star[dat.original$lastC.star<dat.original$time]<-0
  dat.original$C[dat.original$lastC<dat.original$time]<-0
  dat.original$A[dat.original$lastC<dat.original$time]<-0
  dat.original$D[dat.original$lastC<dat.original$time]<-0
  dat.original$censored.star<-0
  dat.original$censored.star[dat.original$lastC.star<dat.original$time]<-1
  dat.original$censored<-0
  dat.original$censored[dat.original$lastC<dat.original$time]<-1
  
  alldata<-dat.original
  alldata$C.star[alldata$lastC.star<alldata$time]<-0
  alldata$A.star[alldata$lastC.star<alldata$time]<-0
  alldata$D.star[alldata$lastC.star<alldata$time]<-0
  alldata$C[alldata$lastC<alldata$time]<-0
  alldata$A[alldata$lastC<alldata$time]<-0
  alldata$D[alldata$lastC<alldata$time]<-0
  
  alldata$CFAR_PID<-paste("A",formatC(as.numeric(alldata$ID),width=6,format='f',digits=0,flag='0'),sep="")
  
  if(!dir.exists('./SurvivalData/Output')){system('mkdir ./SurvivalData/Output')}
  
  if(!dir.exists(paste0('./SurvivalData/Output/', 
                        beta.X, "_", beta.Y, "_",
                        gamma.X, "_", gamma.Y, "_",
                        covXY))){system(paste0('mkdir ./SurvivalData/Output/', 
                                               beta.X, "_", beta.Y, "_",
                                               gamma.X, "_", gamma.Y, "_",
                                               covXY))}
  save(alldata, file=paste0('./SurvivalData/Output/', 
                            beta.X, "_", beta.Y, "_",
                            gamma.X, "_", gamma.Y, "_",
                            covXY, '/SurvivalData_', digit, '.RData'),compress = 'xz')
  
}

# Dr Giganti's code by default generates discretized data, to fit our use, we have to undiscretize it first.
generateUnDiscretizeData <- function(dat){
  
  dat$AGE_AT_LAST_VISIT.star <- dat$lastC.star * 30.437/365.25
  dat$AGE_AT_LAST_VISIT <- dat$lastC * 30.437/365.25
  
  # Find the first A.star and D.star for all the unvalidated measures to obtain the T_0 and T_E
  FirstARTdat <- data.frame(setDT(dat[dat$A.star == 1, c("CFAR_PID", "time")])
                            [, lapply(.SD, min), by = dat[dat$A.star == 1, c("CFAR_PID", "time")]$CFAR_PID])
  
  colnames(FirstARTdat) <- c("CFAR_PID", "FirstARTmonth.star")
  # Join value of 101 to the data for IDs that all the A.star = 0.
  ind_Astar <- which(table(dat$CFAR_PID, dat$A.star)[, 2] == 0)
  FirstARTdat <- rbind(FirstARTdat, data.frame(CFAR_PID = unique(dat$CFAR_PID)[ind_Astar], FirstARTmonth.star = 101))
  
  FirstARTdat.valid <- data.frame(setDT(dat[dat$A == 1 & !is.na(dat$A), c("CFAR_PID", "time")])
                                  [, lapply(.SD, min), by = dat[dat$A == 1 & !is.na(dat$A), c("CFAR_PID", "time")]$CFAR_PID])
  colnames(FirstARTdat.valid) <- c("CFAR_PID", "FirstARTmonth")
  
  FirstOIdat <- data.frame(setDT(dat[dat$D.star == 1, c("CFAR_PID","time")])[, lapply(.SD, min), by = dat[dat$D.star == 1, c("CFAR_PID","time")]$CFAR_PID])
  colnames(FirstOIdat)<-c("CFAR_PID", "FirstOImonth.star")
  
  FirstOIdat.valid <- data.frame(setDT(dat[dat$D == 1 & !is.na(dat$D), c("CFAR_PID","time")])[, lapply(.SD, min), by = dat[dat$D == 1 & !is.na(dat$D), c("CFAR_PID","time")]$CFAR_PID])
  colnames(FirstOIdat.valid)<-c("CFAR_PID", "FirstOImonth")
  
  CensorTimedat<-data.frame(setDT(dat[c("CFAR_PID","time")])[, lapply(.SD, max), by = dat[,c("CFAR_PID","time")]$CFAR_PID])
  colnames(CensorTimedat)<-c("CFAR_PID","maxtime")
  
  Dat1.valid <- join(FirstARTdat.valid, FirstOIdat.valid, type="full")
  Dat2.valid <- join(Dat1.valid, CensorTimedat, type="full")		
  
  Dat1.star <- join(FirstARTdat, FirstOIdat, type="full")
  Dat2.star <- join(Dat1.star, CensorTimedat, type="full")		
  
  cc <- data.table(Dat2.star, key = c("CFAR_PID", "FirstARTmonth.star"))
  dd <- data.table(dat[, c("CFAR_PID", "time", "X", "Y", "AGE_AT_LAST_VISIT.star", 
                           "AGE_AT_LAST_VISIT", "A.star", "A", "C.star", "C", "D", 
                           "lastC.star", "lastC")], 
                   key = c("CFAR_PID", "time"))
  colnames(dd)[colnames(dd) == "time"] <- "FirstARTmonth.star"
  dd$FirstARTmonth.star[which(dd$FirstARTmonth.star == 100)[ind_Astar]] <- 101
  Dat3.star <- merge(cc, dd, by = c("CFAR_PID", "FirstARTmonth.star"))
  ff <- data.table(Dat3.star, key = c("CFAR_PID", "FirstOImonth.star"))
  gg <- data.table(dat[, c("CFAR_PID","time","D.star")], key = c("CFAR_PID","time"))
  colnames(gg) <- c("CFAR_PID", "FirstOImonth.star", "D.star" )
  
  Dat4.star <- gg[ff,]
  Dat4.star <- data.frame(Dat4.star)
  
  Dat4.star$FirstARTmonth.star <- with(Dat4.star, ifelse(is.na(FirstARTmonth.star), 101, FirstARTmonth.star))
  Dat4.star$FirstOImonth.star <- with(Dat4.star, ifelse(is.na(FirstOImonth.star), 101, FirstOImonth.star))
  Dat4.star$ARTage.star <- (Dat4.star$FirstARTmonth.star - 0) * 30.437/365.25
  Dat4.star$OIage.star <- Dat4.star$FirstOImonth.star * 30.437/365.25
  
  Dat4.star$D.star <- with(Dat4.star, ifelse(is.na(D.star), 0, ifelse(FirstOImonth.star == 101, 0, 1)))

  d <- Dat4.star
  d <- merge(d, Dat2.valid)
  d$FirstOImonth <- with(d, ifelse(is.na(FirstOImonth), 101, FirstOImonth))
  d$FirstARTmonth <- with(d, ifelse(is.na(FirstARTmonth), 101, FirstARTmonth))
  d$OIage <- with(d, FirstOImonth * 30.437/365.25)
  d$ARTage <- with(d, (FirstARTmonth - 0) * 30.437/365.25)
  
  d$last.age.star <- d$AGE_AT_LAST_VISIT.star + 30.437/365.25
  d$last.age <- d$AGE_AT_LAST_VISIT + 30.437/365.25
  
  d$ade.star <- with(d, ifelse(OIage.star == (101 * 30.437/365.25), 0, 1))
  d$fu.star <- with(d, ifelse(OIage.star == (101 * 30.437/365.25), last.age.star - ARTage.star, OIage.star - ARTage.star))
  d$fu.star <- with(d, ifelse(fu.star < 0, 0, fu.star))
  d$fu.star <- round(d$fu.star, 3)
  
  d$ade <- with(d, ifelse(OIage == (101 * 30.437/365.25), 0, 1))
  d$fu <- with(d, ifelse(OIage == (101 * 30.437/365.25), last.age - ARTage, OIage - ARTage))
  d$fu <- with(d, ifelse(fu < 0, 0, fu))
  d$fu <- round(d$fu, 3)
  
  return(d)
}












