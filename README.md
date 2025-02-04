<p>Certainly! Here's the README in plain text so you can easily copy and paste it into your GitHub repository's README.md file.</p>
<hr>
<h1>Tower of Hanoi Genetic Algorithm and Simulated Annealing</h1>
<p>Welcome to the Tower-of-Hanoi_GA_SA repository, which contains implementations of Genetic Algorithm (GA), Nested Genetic Algorithm (Nested GA), and Simulated Annealing (SA) for solving the Tower of Hanoi problem. This project was developed as part of a Master's thesis.</p>
<h2>Overview</h2>
<p>The Tower of Hanoi problem is a classic puzzle involving three rods and a number of disks on one rod, each of different sizes, which can slide onto any rod. The puzzle starts with the disks neatly stacked in ascending order of size on one rod, the smallest at the top, making a conical shape.</p>
<p>The algorithms implemented in this project aim to find the optimal sequence of moves to transfer all disks from the initial rod to the goal rod, adhering to the game rules:</p>
<ol>
<li>Only one disk can be moved at a time.</li>
<li>Each move consists of taking the top disk from one of the stacks and placing it on top of another stack.</li>
<li>No disk may be placed on top of a smaller disk.</li>
</ol>
<h2>Structure of the Repository</h2>
<ul>
<li><code>main_ga.py</code>: Main script to run the Genetic Algorithm and Nested Genetic Algorithm.</li>
<li><code>main_sa.py</code>: Main script to run the Simulated Annealing algorithm.</li>
<li><code>README.md</code>: Description and documentation of the repository.</li>
</ul>
<h2>Requirements</h2>
<p>Ensure you have Python installed (version 3.6 or higher is recommended). The project depends on several Python libraries:</p>
<ul>
<li><code>numpy</code></li>
<li><code>copy</code></li>
</ul>
<p>You can install the required libraries via pip:</p>
<div class="code-block-parent">
<button class="copy-code-button"><i aria-label="Copy code the below code snippet" class="pi pi-copy"></i> Copy code</button>
<div class="code-block-container">
<pre style="overflow-x: auto;"><code class="language-bash">pip install numpy copy
</code></pre>
</div>
</div>
<h2>Running the Scripts</h2>
<h3>Genetic Algorithm</h3>
<p>To run the Genetic Algorithm and Nested Genetic Algorithm, execute <code>main_ga.py</code>:</p>
<div class="code-block-parent">
<button class="copy-code-button"><i aria-label="Copy code the below code snippet" class="pi pi-copy"></i> Copy code</button>
<div class="code-block-container">
<pre style="overflow-x: auto;"><code class="language-bash">python main_ga.py
</code></pre>
</div>
</div>
<h3>Simulated Annealing</h3>
<p>To run the Simulated Annealing algorithm, execute <code>main_sa.py</code>:</p>
<div class="code-block-parent">
<button class="copy-code-button"><i aria-label="Copy code the below code snippet" class="pi pi-copy"></i> Copy code</button>
<div class="code-block-container">
<pre style="overflow-x: auto;"><code class="language-bash">python main_sa.py
</code></pre>
</div>
</div>
<h3>Example Usage</h3>
<p>You may modify parameters such as the number of disks, maximum iterations, population size, etc., directly in the script files to suit your experimental needs.</p>
<h2>Methods and Results</h2>
<h3>Genetic Algorithm (GA)</h3>
<p>The GA mimics the process of natural selection. It works with a population of possible solutions and evolves them over several generations using selection, crossover, and mutation operators.</p>
<h3>Nested Genetic Algorithm (Nested GA)</h3>
<p>The Nested GA introduces an additional layer where multiple instances of the GA run iteratively. Each instance generates a subset of solutions, leading to a refined search towards the optimal solution.</p>
<h3>Simulated Annealing (SA)</h3>
<p>SA is inspired by the annealing process in metallurgy. It explores the solution space by probabilistically deciding whether to accept worse solutions as it cools down the temperature parameter, aiming to escape local optima.</p>
<h3>Thesis Results</h3>
<p>The algorithms were tested extensively:</p>
<ul>
<li><strong>3-disks variant:</strong> Nested GA consistently outperformed other methods, finding optimal solutions in all runs.</li>
<li><strong>4-disks variant:</strong> Standard GA found an optimal solution in a few runs and was overall better in mean, standard deviation, minimum, and maximum fitness compared to Nested GA and SA.</li>
<li><strong>5-disks variant:</strong> Standard GA outperformed others, while none of the algorithms managed to find the optimal solution.</li>
</ul>
<p>Statistical analyses confirmed the significance of these results. The conclusion is that while the Nested GA is advantageous in a relatively smaller solution space, the standard GA is more suitable as the complexity increases.</p>
<p>For more detailed results and analysis, please refer to the Results section in the thesis document.</p>
<h2>Contact</h2>
<p>For any questions or feedback, please contact the repository owner at [<a title="" target="_blank" href="mailto:your-email@example.com">your-email@example.com</a>].</p>
<hr>
<p>Feel free to replace <code>[your-email@example.com]</code> with your actual email address. You can now copy and paste this into your GitHub repository's README.md file.</p>
