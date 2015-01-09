# Goal is to identify if the frequency distribution is normal.
# Otherwise, determine what kind of distribution it has.

# A typical frequency distribution measures:
#   1. Frequency
#   2. Relative Frequency
#   3. Cumulative Frequency
#   4. Cumulative Relative Frequency

# TO DO LIST:
#   - DONE - Input a vector
#   - DONE - Output a frequency table where the following measures are calculated for each value
#      Frequency + Relative Frequency + Cumulative Frequency + Cumulative Relative Frequency
#   - DONE - Plot the frequency distribution as a histogram or a line.  Base the chart type on
#      an input.  Base the groupings on an input
#   - Make the cumulative series less bright - more transparent, less intense.
#   - Is there a standard approach to charting frquency distributions?
#   - Provide a way to attach what we learn as an attribute of the vector.  Is this possible?


#  INPUTS:
#  1.) Vector
#  2.) Number of Bins

#  OUTPUTS:
#  1.) Plot

library("qdap")    # used to calculate frequency distributions
library("reshape") # used to convert the freq distributions into a graphable data frame
library("ggplot2") # used to graph our data

# x <- c(1,2,3,2,4,2,5,4,6,7,8,9)

freqDistrib <- function(inputVector, parmBreaks=6) {

	df <- dist_tab(inputVector, breaks=parmBreaks)
	df.melt <- melt(df, id.vars="interval", measure.vars=c("Freq", "cum.Freq"))
	df.melt.percents <- melt(df, id.vars="interval", measure.vars=c("percent", "cum.percent"))
	percents <- df.melt.percents$value
	df.melt <- cbind(df.melt, percents)
	df.melt$Offset <- -0.3
	z <- which(df.melt[,2]=="Freq")
	df.melt[z,]$Offset <- 1.5

	ggplot(df.melt, aes(x=interval, y=value, fill=variable, title='Frequency Distribution')) + 
	  geom_bar(stat="identity", position="dodge") + geom_text(aes(label=paste(percents,"%",sep="")), position=position_dodge(width=0.9), vjust=3, size=3)
	}

# freqDistrib(x)

df <- dist_tab(x, breaks=4)
df.melt <- melt(df, id.vars="interval", measure.vars=c("Freq", "cum.Freq"))
df.melt.percents <- melt(df, id.vars="interval", measure.vars=c("percent", "cum.percent"))
percents <- df.melt.percents$value
df.melt <- cbind(df.melt, percents)
df.melt$Offset <- -0.3
z <- which(df.melt[,2]=="Freq")
df.melt[z,]$Offset <- 1.5
str(df.melt)

ggplot(df.melt, aes(x=interval, y=value, fill=variable, title='Frequency Distribution')) + 
  geom_bar(stat="identity", position="dodge") + geom_text(aes(label=percents), position=position_dodge(width=0.9), vjust=3, size=3)

help(which)
help(geom_text)
help(ggplot)
help(melt)
help(join)
help(merge)
help(geom_bar)
help(dist_tab)
