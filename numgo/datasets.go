package numgo

import (
	"math"
	"math/rand"
)

func GetSpiral(train bool)([][]float64,[]int){
	if (train){
		rand.Seed(1984)
	}else{
		rand.Seed(2020)
	}
	numData,numClass:=100,3
	dataSize:=numData*numClass
	x:=make([][]float64,dataSize)
	t:=make([]int,dataSize)
	for j :=0;j<numClass;j++{
		nd:=float64(numData)
		for i:=0;i<numData;i++{
			rate:=float64(i)/nd
			radius:=1.0*rate
			theta:=float64(j)*4.0+4.0*rate+rand.NormFloat64()*0.2
			ix:=numData*j+i
			xv := []float64{radius * math.Sin(theta), radius * math.Cos(theta)}
			x[ix]= xv
			t[ix]=j //分类标签
		}
	}
	//shuffle
	rand.Shuffle(len(x), func(i, j int) {
		x[i],x[j]=x[j],x[i]
		t[i],t[j]=t[j],t[i]
	})
	return x,t
}
