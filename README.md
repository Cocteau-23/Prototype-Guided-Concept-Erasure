# Prototype-Guided Concept Erasure in Diffusion Models
---

## Installation

---

### Project Organization

```latex
Prototype-Guided Concept Erasure
	|---data
			(directory for prompts dataset and generated prompts)
	|---eval
			(directory for evaluate scripts)
	|---output
			(directory for experiment results and generated images)
	|---prototypes
			(directory for prototypes)
	|---src
			(directory for experiment scripts)
```

### Environment Installation

```latex
conda create -n proto_ce python=3.12
concda activate proto_ce
pip install -r requirements.txt
```

## Run

---

We have provided a script that covers the entire process from generating prompts, generating sample images, training, and erasing. 

- When the concept to be erased is a concept in the I2P, the relevant concept name can be directly used as categories. When the concept to be erased is IP or style, the corresponding concept_facets in schema.json need to be modified.

```latex
bash run.sh
```
