package main

import "fmt"

type Probability struct {
	PriorProb       float64
	ConditionalProb float64
	EvidenceProb    float64
}

func (p *Probability) CalculatePosteriorProb() float64 {
	// Menghitung posterior probability menggunakan rumus Bayes
	posteriorProb := (p.ConditionalProb * p.PriorProb) / p.EvidenceProb
	return posteriorProb
}

const (
	PDiagnosedGivenHIV      = 0.95 // P(D|S)
	PNotDiagnosedGivenNoHIV = 0.98 // P(-D|-S)
	PDiagnosedGivenNoHIV    = 0.02 // P(D|¬S)
	PNotDetectedGivenSpam   = 0.05 // P(-D|S)
)

func TotalProbability(prior *float64) float64 {
	// using law of total probability.
	// P(D) = P(D|S)P(S) + P(D|¬S)P(¬S)
	PNoHIV := 1 - *prior
	return (PDiagnosedGivenHIV * *prior) + (PDiagnosedGivenNoHIV * PNoHIV)
}

func main() {
	prior := 0.03
	evidence := TotalProbability(&prior)

	prob := Probability{
		PriorProb:       prior,    // Probabilitas prior
		ConditionalProb: 0.95,     // Probabilitas kondisional
		EvidenceProb:    evidence, // Probabilitas evidence
	}

	// Menghitung posterior probability menggunakan Bayes' Rule
	posteriorProb := prob.CalculatePosteriorProb()

	fmt.Printf("Posterior Probability: %.2f\n", posteriorProb)
	fmt.Println(" ")

}
