package numgo

import (
	"math"
	"math/rand"

	nd "test_ai/numed"
	ut "test_ai/utils"
)
type IDataset interface {
	prePare(data *dataSet)
}
type dataSet struct {
	Train bool
	Data *nd.NumEd
	Label []int
	IDataset IDataset
	trans *Compose
	targetTrans *Compose
}

func (d *dataSet) Init( ) {
	d.IDataset.prePare(d)
}
func (d *dataSet) Len( ) int {
	return len(d.Label)
}
func (d *dataSet) Get(index int) ([]float64,int){
	data := d.Data.RowData(index)
	////todo debug image show
	//pv := nd.NewVec(data...)
	//if pv.Sum(nil,true).Var()>1{
	//	PrintImg(data,d.Label[index])
	//}

	d.trans.Run(data)
	if d.Label==nil{
		return data, 0
	}else{
		label := d.Label[index]
		return data, label
	}
}

type spiral struct {
	dataSet
}

func (s spiral) prePare(d *dataSet) {
	data,label:=GetSpiral(s.Train)
	d.Data=nd.NewDense(ut.Flatten(data))
	d.Label =label
}

func Spiral(train bool) *dataSet {
	d:= dataSet{IDataset: &spiral{dataSet{Train: train}}}
	d.Init()
	return &d
}

func GetSpiral(train bool)([][]float64,[]int){
	if (train){
		rand.Seed(1984)
	}else{
		rand.Seed(2020)
	}
	numData,numClass:=100.0,3.0
	dataSize:=int(numData*numClass)
	x:=make([][]float64,dataSize)
	t:=make([]int,dataSize)
	for j :=0.0;j<numClass;j++{
		for i:=0.0;i<numData;i++{
			rate:=i/ numData
			radius:=1.0*rate
			theta:=j*4.0+4.0*rate+rand.NormFloat64()*0.2
			ix:=int(numData*j+i)
			xv := []float64{radius * math.Sin(theta), radius * math.Cos(theta)}
			x[ix]= xv
			t[ix]=int(j) //分类标签
		}
	}
	//shuffle
	rand.Shuffle(len(x), func(i, j int) {
		x[i],x[j]=x[j],x[i]
		t[i],t[j]=t[j],t[i]
	})
	return x,t
}

