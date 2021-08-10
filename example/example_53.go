package main

import "C"
import (
	"fmt"
	"os"

	ng "test_ai/numgo"
)

func main() {
	maxEpoch := 3
	batchSize := 100
	opt := ng.SGD(1.0)
	//opt:=ng.Adam(0.01,0.9,0.999,1e-8)
	model := ng.MLP(ng.ReLU, opt, 100, 10)
	trainSet := &ng.DataSet{}
	trainSet = ng.Mnist(true)
	trainLen := float64(trainSet.Len())
	trainLoader := ng.NewDataLoader(trainSet, batchSize, true)
	weightName := "my_mlp.gob"
	if _, err := os.Stat(weightName); !os.IsNotExist(err) {
		model.LoadWeights(weightName)
	}
	for i := 0; i < maxEpoch; i++ {
		ng.Backprop = true
		sumLoss, sumAcc := 0.0, 0.0
		for trainLoader.HasNext() {
			dx, dt := trainLoader.Next()
			y := model.Forward(dx)
			loss := ng.SoftmaxCrossEntroy(y, dt)
			acc := ng.Accuracy(y, dt)
			model.ClearGrad()
			loss.Backward(false)
			model.Grad2Param()
			l := float64(len(dt))
			sumLoss += loss.Var() * l
			sumAcc += acc.Var() * l
			fmt.Printf("train loss %.2f ,accuracy:%.2f\n", sumLoss/trainLen, sumAcc/trainLen)

		}
		fmt.Printf("epoch %d \n", i+1)
		fmt.Printf("train loss %.2f ,accuracy:%.2f\n", sumLoss/trainLen, sumAcc/trainLen)
	}
	model.SaveWeights(weightName)
}
