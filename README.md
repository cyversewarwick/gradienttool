# Gradient Tool

## The Purpose of the Algorithm

In the case of some experiments, it is impossible to create a suitable control. A good example would be the study of ageing. The gradient tool is a method designed for the detection of expression trends (up/down-regulation) in such settings where only a single condition is available.

## Basic Input/Output

The gradient tool accepts expression data formatted into a CSV on input, with the first column being gene names and the first row being time point information. The output features detailed information on the resulting gradient, with one row per time point per gene, and the last column featuring a z-score that captures the extremity of the up/down-regulation at the given time point (the further away from 0, the more extreme the trend). These can be converted into a simple list of times of first significant change based on a threshold value by the helper `GradientTool_Threshold` app detailed at the end of the documentation.

## How Does It Work?

The gradient tool begins by fitting a Gaussian process to all of the data for a single gene, then follows it up by computing a derivative of the obtained fit. The resulting gradient allows for the quantification of regulatory trends present in the original Gaussian process fit, as the gradient mean and variance can be used to compute explicit z-scores. The further away from 0 the score, the more extreme the regulatory phenomenon. For details, consult [Breeze et al., 2011][breeze2011].

## Test Run

If you want to get a feel for input and output formatting, basic demo data is provided at `ktpolanski/gradienttool_testdata/input.csv` under Community Data.

## Input In Detail

### Expression CSV File

**Mandatory input.** Expression data to be analysed with the gradient tool. First column for gene names, first row for time point information.

### Normalise Data

**Default:** Checked (Yes)

Checking the box will have the algorithm normalising the provided data, whilst unchecking it will have it run on the data as provided without any processing.

## Output In Detail

### `out.csv`

Detailed output of the resulting gradient, with one line per time point (second column) per gene (first column). Information on the mean and variance of the original Gaussian process fit and the gradient are contained in subsequent columns, with the final column being the z-score of the gradient at that point. The further away from 0 the value, the more likely that a regulatory phenomenon is happening at that point.

### `out.pdf`

A PDF file containing the visualisation of the obtained fits. Each gene features two plots - one showing the original Gaussian process fit, and the second one displaying the gradient obtained from it.



# `Gradienttool_Filter`

## The Purpose of the Algorithm

The gradient tool output, while comprehensive, could use a more concise form when trying to quickly list genes that are identified as up/down-regulated. This helper app takes the output produced by the gradient tool and does that.

## Test Run

If you want to see what the output of this helper app looks like, you can find the test data at `ktpolanski/gradienttool_filter_testdata/out.csv` under Community Data. `out.pdf` is also provided to give a complete set of gradient tool outputs and a visual feel of the fits, but it's not to be used on input by the app.

## Input

### Gradient Tool Output

**Mandatory input.** The CSV output of the gradient tool app.

### Z-Score Threshold

**Default:** 2

The threshold of the z-score to declare that a regulatory phenomenon is in place. If the z-score value for a gene is above this value for the first time, it is reported in the output. If the z-score value for a gen is below the negative of this value for the first time, it is reported in the output. The minimum of those two values is also reported as the time of the first change. In case a gene is only up/down-regulated during the experiment, the time of the other regulatory change will be set to `nan`.

## Output

### `ChangingGenes.txt`

A more concise version of the output produced by the gradient tool, with the provided z-score threshold applied to produce a time of first up-regulation, time of first down-regulation and time of first change. Can be used on input of the Expression Filter app, as it's tab-delimited and all genes that don't have any change during the course of the experiment are removed from the output.

[breeze2011]: http://www.plantcell.org/content/23/3/873.full