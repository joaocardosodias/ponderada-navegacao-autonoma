"""Setup script for the cg_navigator package."""

from setuptools import find_packages, setup

package_name = 'cg_navigator'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='João Cardoso Dias',
    maintainer_email='joao.dias@sou.inteli.edu.br',
    description='Autonomous maze navigation and mapping for Culling Games',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'part1 = cg_navigator.part1_navigator:main',
            'part2 = cg_navigator.part2_mapper:main',
        ],
    },
)
