package main

import "C"
import (
	"fmt"
	"math"

	nd "test_ai/numed"
	ng "test_ai/numgo"
	ut "test_ai/utils"
)

func main() {
	maxEpoch := 30
	batchSize := 30
	hiddenSize := 20
	bpttLength := 30
	trainSet := ng.SinCurve(true)
	trainLoader := ng.NewSeqDataLoader(trainSet, batchSize)
	seqlen := trainSet.Len()

	opt := ng.Adam(0.01, 0.9, 0.999, 1e-8)
	model := ng.NewBetterRNN(hiddenSize, 1, opt)
	for i := 0; i < maxEpoch; i++ {
		model.ResetState()
		loss, lossV, count := ng.NewVar(0), 0.0, 0
		for trainLoader.HasNext() {
			dx, dt := trainLoader.Next()
			y := model.Forward(dx)
			dv := ng.NewVec(dt...).ReShape(y.Data.Dims())
			dv.Name = "label"
			loss = ng.MeanSquaredError(y, dv)
			lossV += loss.Var()
			//loss = ng.Add(loss, ng.MeanSquaredError(y, dv))

			count += 1
			//达到BPTT或者数据末尾
			if count%bpttLength == 0 || count == seqlen {
				model.ClearGrad()
				loss.Backward(false)
				loss.UnchainBackward()
				model.Grad2Param()
			}
		}
		if i == 0 {
			loss.Plot(true, "./temp/lstm.png")
		}
		avgLoss := lossV / float64(count)
		fmt.Printf("epoch %d loss:%f\n", i+1, avgLoss)
	}

	//Plot
	ng.Backprop = false
	xs := nd.NewLinespace(0, 4*math.Pi, 1000).Cos()
	_, xc := xs.Dims()
	model.ResetState()

	xf, predList := make([]float64, xc), make([]float64, xc)
	for i := 0; i < xc; i++ {
		x := xs.Get(0, i)
		y := model.Forward(x)
		xf[i], predList[i] = x, y.Var()
	}

	pl := ng.NewPlot("LSTM", "X", "Y", 200, 150)
	pl.Plot("sin", ut.Arange(0, float64(xc), 1), xf, nil)
	pl.Plot("pred", ut.Arange(0, float64(xc), 1), predList, nil)
	pl.Save("./temp/points.png")

}
