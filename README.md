<p align="center"><h1 align="center">IMAGELAND</h1></p>
<p align="center">
	<em>a physics-based image processing toolkit</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/jaccl17/imageland?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/jaccl17/imageland?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/jaccl17/imageland?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/jaccl17/imageland?style=default&color=0080ff" alt="repo-language-count">
</p>
<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Prerequisites](#-prerequisites)
  - [ Installation](#-installation)
  - [ Testing](#-testing)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)

---

##  Overview
**<em>imageland</em>** is the repository for the image-processing tools I have designed. What you see is the underdeveloped and unspecialized versions of each tool, which can be adjusted and editted as needed for your own explorations. Most tools use mathematics to transform and filter 2D images. It is recommended to be familiar with fourier transforms, mode decomposition, and neural networks to understand some of the tools.


--- 

##  Features

**Neural Networks (/nn):** Here I have placed basic examples of image classification neural networks, including data sorting, in the form of a '.py' and '.ipynb'

**Pseudo-Bi-directional Empirical Mode Decomposition (/BEMD):** This is a applies a 1D Empircal Mode Decomposition to the rows of a 2D array and then to the column of the decomposed matrix - this can be scaled to higher N-dimensional matrices. I include a BEMB library-based functionality as well as a my own BEMB process. More information can be found <a href="https://en.wikipedia.org/wiki/Multidimensional_empirical_mode_decomposition">here</a> 

**Discrete Cosine Transformation (/DCT):** This transformation operates on the entire 2D array at once and deconstructs the row and column signals into a series of cosine functions. From this cosine space, one can filter an image by feature angle (0-90deg) as well as scale (px-Mpx). More information can he found here <a href="https://en.wikipedia.org/wiki/Discrete_cosine_transform">here</a>

**Fast Fourier Transform (/FFT):** This type of transformation is similar to DCT in that the 2D array is mapped into frequency space, however the FFT allows for filtering based on phase as well as magnitude. More information can he found here <a href="https://en.wikipedia.org/wiki/Fast_Fourier_transform">here</a> 

**Image Stacking (/image_stacking):** This is a very simple script that performs stacking by mean, median, or weighted mean on a series of images.

---


##  Project Structure

```sh
└── imageland/
    ├── BEMD
    │   ├── BEMD_scratch.ipynb
    │   ├── BEMD_test.ipynb
    │   ├── BEMD_test.py
    │   └── BEMD_test2.py
    ├── DCT
    │   ├── dct_official.py
    │   ├── dct_test.ipynb
    │   ├── dct_test.py
    │   └── dct_test_bandfilter.py
    ├── Desktop
    │   └── images_capture
    ├── FFT
    │   ├── fft_filter.py
    │   ├── fft_filter_test.py
    │   └── mask_generator_test.ipynb
    ├── README.md
    ├── downsize_calculator.py
    ├── image_cropping.py
    ├── image_stacking
    │   └── stacking_test.ipynb
    ├── nn
    │   ├── directory_split.py
    │   ├── nn1.ipynb
    │   └── nn1.py
    ├── popcorn_stand.py
    └── test.py
```


###  Project Index
<details open>
	<summary><b><code>IMAGELAND/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/image_cropping.py'>image_cropping.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/downsize_calculator.py'>downsize_calculator.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/test.py'>test.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/popcorn_stand.py'>popcorn_stand.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- nn Submodule -->
		<summary><b>nn</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/nn/directory_split.py'>directory_split.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/nn/nn1.py'>nn1.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/nn/nn1.ipynb'>nn1.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- image_stacking Submodule -->
		<summary><b>image_stacking</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/image_stacking/stacking_test.ipynb'>stacking_test.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- DCT Submodule -->
		<summary><b>DCT</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/DCT/dct_test.py'>dct_test.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/DCT/dct_test.ipynb'>dct_test.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/DCT/dct_official.py'>dct_official.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/DCT/dct_test_bandfilter.py'>dct_test_bandfilter.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- BEMD Submodule -->
		<summary><b>BEMD</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/BEMD/BEMD_scratch.ipynb'>BEMD_scratch.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/BEMD/BEMD_test.py'>BEMD_test.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/BEMD/BEMD_test.ipynb'>BEMD_test.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/BEMD/BEMD_test2.py'>BEMD_test2.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- FFT Submodule -->
		<summary><b>FFT</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/FFT/mask_generator_test.ipynb'>mask_generator_test.ipynb</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/FFT/fft_filter.py'>fft_filter.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/jaccl17/imageland/blob/master/FFT/fft_filter_test.py'>fft_filter_test.py</a></b></td>
				<td><code>❯ REPLACE-ME</code></td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---
##  Getting Started

###  Prerequisites

Before getting started with imageland, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python
- **Package Manager:** Poetry, Pip, Conda


###  Installation

Install imageland using one of the following methods:

**Build from source:**

1. Clone the imageland repository:
```sh
❯ git clone https://github.com/jaccl17/imageland
```

2. Navigate to the project directory:
```sh
❯ cd imageland
```

3. Edit and run scripts from terminal or IPE:




###  Testing
All explorations have their own folder. Each folder contains a test script and/or graphing tools for troubleshooting

---
##  Project Roadmap

- [X] **`Task 1`**: <strike>Develop repo to organize scripts.</strike>
- [X] **`Task 2`**: <strike>Comment and polish scripts (organize into code chunks).</strike>
- [ ] **`Task 3`**: Combine tools into a single program.
- [ ] **`Task 4`**: Build downloadable UI.

---

##  Contributing

- **💬 [Join the Discussions](https://github.com/jaccl17/imageland/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://github.com/jaccl17/imageland/issues)**: Submit bugs found or log feature requests for the `imageland` project.
- **💡 [Submit Pull Requests](https://github.com/jaccl17/imageland/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/jaccl17/imageland
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://github.com{/jaccl17/imageland/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=jaccl17/imageland">
   </a>
</p>
</details>

---
