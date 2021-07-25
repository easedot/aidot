package numgo

import (
	"math/rand"

	nd "test_ai/numed"
	ut "test_ai/utils"
)

func NewDataLoader(d *dataSet,batchSize int,shuffle bool) *DataLoader {
	maxIter:=d.Len()/batchSize
	dr := &DataLoader{dataSet: d, batchSize: batchSize, shuffle: shuffle, maxIter: maxIter}
	dr.reset()
	return dr
}

type DataLoader struct {
	dataSet   *dataSet
	batchSize int
	dataSize  int
	gpu       bool
	shuffle   bool
	maxIter   int
	iteration int
	index     []int
}

func (d *DataLoader) reset() {
	d.iteration=0
	if d.shuffle{
		d.index=rand.Perm(d.dataSet.Len())
	}else{
		d.index=ut.ArangeInt(0,d.dataSet.Len(),1)
	}
}
func (d *DataLoader) HasNext() bool {
	if d.iteration<d.maxIter{
		return true
	}else{
		d.reset()
		return false
	}
}
func (d *DataLoader) Next() (x *nd.NumEd,t []int){
	batchIndex:=d.index[d.iteration*d.batchSize :(d.iteration+1)*d.batchSize]
	xf,tf:=make([][]float64, d.batchSize),make([]int, d.batchSize)
	for i,r:=range batchIndex{
		xt,tt:=d.dataSet.Get(r)
		xf[i]=xt
		tf[i]=tt
	}
	d.iteration+=1
	return nd.NewDense(ut.Flatten(xf)),tf
}
func (d *DataLoader) ToGpu() {
	d.gpu=true
}
func (d *DataLoader) ToCpu() {
	d.gpu=false
}