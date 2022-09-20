============
User Notices
============
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :target: http://opensource.org/licenses/MIT

.. warning::
	The original version of **MoSDeF-GOMC**, which uses **Parmed** as the software backend, will be deprecated by December 2022.  This **Parmed** version is already replaced with the new **MoSDeF-GOMC** version, which uses **MoSDeF's GMSO** software as the new backend.  We recommend that the new **GMSO** version of **MoSDeF-GOMC** be used because it has many new features, and the **Parmed** is no longer supported.

The new **MoSDeF-GOMC** version using GMSO allows for a more flexible and better user experience as it has the following new features:

	#. The ability to use Lennard-Jones (LJ) and the Mie potential individually or in combination for non-bonded interactions.  The Exp6 (Buckingham) non-bonded potential will be added in the future.

	#. The user can build a force field file using the **GMSO** formatted XML, which allows the user to enter the equations and units for the non-bonded and bonded interactions. This includes different dihedral forms (RB-torsions, OPLS, and periodic dihedrals) and the ability to use the Kelvin energy units. Provided the equation form is permitted in the **MoSDeF-GOMC** software, **MoSDeF-GOMC** automatically converts them to the usable form and scales the coefficients accordingly

	#. All **MoSDeF-GOMC** functions now require the `unyt <https://unyt.readthedocs.io/en/stable/>`_ package units unless the values are unitless, ensuring the user enters the proper units.   **MoSDeF-GOMC** will internally handle the unit conversions and write the files with the correct units.

	#. Automatic application of the mixing/combining rules, if provided in the force field XML files.

	#. Other new features exist, so please see the **MoSDeF-GOMC** documentation for more features.


