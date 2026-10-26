# Reverse-engineering workspace

Notes, scripts and evidence from reverse engineering Apple's `jax-metal` 0.1.1 plugin with
REA (Reverse Engineer Anything) and Ghidra; see `FINDINGS.md` and `../PLAN.md`. The project
started from this brief:

> We are going to use Reverse Engineer Everything to look into the abandoned jax-metal plugin to try to get JAX to work
> on a modern MacOS system and GPU. The information on JAX-metal can be found here: https://developer.apple.com/metal/jax/.
> The actual JAX repository is available here: https://github.com/jax-ml/jax. All development should take place in the
> ~/jax_metal directory, which contains this README, and is a git repository.
>
> The information for installing reverse engineer everything is available here: https://reverseengineeranything.app/get-started/.
> The node requirement has already been installed.
> Install REA and connect it to this coding agent using npx rea-agents@latest setup. Show me the setup plan for approval, then verify the installation.
> Then, install the existing jax-metal program where it can be connected to REA.
> Finally, present your plan of attack for using REA to reverse engineer jax-metal, with an estimate of the degree of complexity of the problem.
