# Repo Template

Kick off a project with the right foot.

A repository template for easily setting up a well behaved development environment for a smooth
collaboration experience.

This template takes care of setting up and configuring:

- A **virtual environment**
- **Formatting and linting** tools
- Some shared default **VSCode settings**
- A **Pull Request template**
- A **GitHub Action** that runs formatting and linting checks

Any of these configurations and features can be disabled/modified freely after set up if the team
chooses to.

Note: [pyenv](https://github.com/pyenv/pyenv#installation) and
[poetry](https://python-poetry.org/docs/#installation) are used for setting up a virtual environment
with the correct python version. Make sure both of those are installed correctly in your machine.

# Usage

1. Click the `Use this template` button at the top of this repo's home page to spawn a new repo
   from this template.

2. Clone the new repo to your local environment.

3. Run `sh init.sh <your_project_name> <python version>`.

   Note that:

   - the project's accepted python versions will be set to `^<python version>` - feel free
     to change this manually in the `pyproject.toml` file after running the script.
   - your project's source code should be placed in the newly-created folder with your project's
     name, so that absolute imports (`from my_project.my_module import func`) work everywhere.

4. Nuke this readme and the `init.sh` file.

5. Add to git the changes made by the init script, such as the newly created `poetry.toml`,
   `poetry.lock` and `.python-version` files.

6. Commit and push your changes - your project is all set up.

7. [Recommended] Set up the following in your GitHub project's `Settings` tab:
   - Enable branch protection for the `main` branch in the `Branches` menu to prevent non-reviewed
     pushes/merges to it.
   - Enable `Automatically delete head branches` in the `General` tab for feature branches to be
     cleaned up when merged.

# For ongoing projects

If you want to improve the current configs of an existing project, these files are the ones you'll
probably want to steal some content from:

- [VSCode settings](.vscode/settings.json)
- [Flake8 config](.flake8)
- [Black and iSort configs](pyproject.toml)
- [Style check GitHub Action](.github/workflows/style-checks.yaml)

Additionally, you might want to check the
[project's source code is correctly installed via Poetry](https://stackoverflow.com/questions/66586856/how-can-i-make-my-project-available-in-the-poetry-environment)
for intra-project imports to work as expected across the board.

# For developers of this template

To test new changes made to this template:

1. Run the template in test mode with `test=true sh init.sh <your_project_name> <python version>`,
   which will not delete the [project_base/test.py](project_base/test.py) file from the source
   directory.

2. Use that file to check everything works as expected (see details in its docstring).

3. Make sure not to version any of the files created by the script. `git reset --hard` + manually
   deleting the created files not yet added to versioning works, for example.

# Issues and suggestions

Feel free to report issues or propose improvements to this template via GitHub issues or through the
`#team-tech-meta` channel in Slack.

# Can I use it without Poetry?

This template currently sets up your virtual environment via poetry only.

If you want to use a different dependency manager, you'll have to manually do the following:

1. Remove the `.venv` environment and the `pyproject.toml` and `poetry.lock` files.
2. Create a new environment with your dependency manager of choice.
3. Install flake, black and isort as dev dependencies.
4. Install the current project's source.
5. Set the path to your new environment's python in the `python.pythonPath` and
   `python.defaultInterpreterPath` in [vscode settings](.vscode/settings.json).

Disclaimer: this has not been tested, additional steps may be needed.

# Troubleshooting

### pyenv not picking up correct python version from .python-version

Make sure the `PYENV_VERSION` env var isn't set in your current shell
(and if it is, run `unset PYENV_VERSION`).
