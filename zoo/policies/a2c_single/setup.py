from setuptools import setup, find_packages
from a2c_agent.version import VERSION

setup(
    name="a2c-rl-agent",
    description="a2c rl agent inference",
    version=VERSION,
    packages= find_packages(include=[
        "a2c_agent",
        "a2c_agent.A2C"
        "a2c_agent.risk_indices",
    ]),
    include_package_data=True,
    install_requires=["torch", "smarts"],
)
