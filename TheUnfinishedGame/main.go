package main

import (
	"fmt"
)

// type Dist map[string]int

// type Dist map[string]int

// func favorable(event map[string]bool, space []string) Dist {
//     dist := make(Dist)
//     for _, outcome := range space {
//         if event[outcome] {
//             dist[outcome]++
//         }
//     }
//     return dist
// }


// func (d Dist) Cases() int {
// 	total := 0
// 	for _, v := range d {
// 		total += v
// 	}
// 	return total
// }

// func (d Dist) Favorable(event map[string]bool) Dist {
// 	fav := Dist{}
// 	for k, v := range d {
// 		if event[k] {
// 			fav[k] = v
// 		}
// 	}
// 	return fav
// }

// func P(event map[string]bool, space []string) float64 {
// 	fav := Dist(favorable(event, space))
// 	eventCases := fav.Cases()
// 	spaceCases := Dist(space).Cases()
// 	return float64(eventCases) / float64(spaceCases)
// }

func main() {
	// space := []string{"HH", "HT", "TH", "TT"}
	// event := map[string]bool{"HT": true, "TH": true}
	// p := P(event, space)
	// fmt.Println(p)
	fmt.Println("Hello, playground")
}