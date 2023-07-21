package main

import (
	"math"
	"testing"
)

const epsilon = 0.00000001 // Toleransi kesalahan

func TestCalculatePosteriorProb(t *testing.T) {
	prob := Probability{
		PriorProb:       0.1,  // Probabilitas prior
		ConditionalProb: 0.75, // Probabilitas kondisional
		EvidenceProb:    0.1,  // Probabilitas evidence
	}

	expectedPosteriorProb := 0.75

	actualPosteriorProb := prob.CalculatePosteriorProb()

	if math.Abs(actualPosteriorProb-expectedPosteriorProb) > epsilon {
		t.Errorf("Wrong answer! it should be %.8f", expectedPosteriorProb)
	}
}

func TestTotalProbability(t *testing.T) {

	const (
		PDetectedGivenSpam       = 0.95 // P(D|S)
		PNotDetectedGivenNonSpam = 0.98 // P(-D|-S)
		PDetectedGivenNonSpam    = 0.02 // P(D|¬S)
		PNotDetectedGivenSpam    = 0.05 // P(-D|S)
	)

	e := 0.0001
	prior := 0.2

	expectedTotalProb := 0.2060
	actualTotalProb := TotalProbability(&prior)

	if math.Abs(actualTotalProb-expectedTotalProb) > e {
		t.Errorf("Wrong answer! It should be %.4f but %.4f", expectedTotalProb, actualTotalProb)
	}
}
