package main

import "C"
import (
	"fmt"

	ng "test_ai/numgo"
)

func main() {
	maxEpoch := 5
	batchSize := 30
	lr := 1.0
	sgd := ng.SGD(lr)
	model := ng.MLP(ng.Sigmoid, sgd, 100, 10)

	trainSet := ng.Mnist(true)
	trainLen := float64(trainSet.Len())
	testSet := ng.Mnist(false)
	testLen := float64(testSet.Len())

	trainLoader := ng.NewDataLoader(trainSet, batchSize, true)
	testLoader := ng.NewDataLoader(testSet, batchSize, false)

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
			sumLoss += loss.Data.Var() * l
			sumAcc += acc.Data.Var() * l
			fmt.Printf("train loss %.2f ,accuracy:%.2f\n", sumLoss/trainLen, sumAcc/trainLen)

		}
		fmt.Printf("epoch %d \n", i+1)
		fmt.Printf("train loss %.2f ,accuracy:%.2f\n", sumLoss/trainLen, sumAcc/trainLen)

		ng.Backprop = false
		sumLoss, sumAcc = 0.0, 0.0
		for testLoader.HasNext() {
			dx, dt := testLoader.Next()
			y := model.Forward(dx)
			loss := ng.SoftmaxCrossEntroy(y, dt)
			acc := ng.Accuracy(y, dt)
			l := float64(len(dt))
			sumLoss += loss.Data.Var() * l
			sumAcc += acc.Data.Var() * l
		}
		fmt.Printf("test loss %.2f ,accuracy:%.2f\n", sumLoss/testLen, sumAcc/testLen)
	}
}
