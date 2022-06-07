class CountryUndefinedException(Exception):
    """
    Declaration of new RuntimeException to be thrown when an instance of the class EWarningGeneral have any trouble
    finding the desired countries to be studied.
    """

    def __int__(self, message):
        """
        Basic constructor that calls to super constructor of Exception.

        :param string message: Exception message error to be shown.
        """
        super().__init__(message)


class DateOutRangeException(Exception):
    """
    Declaration of new RuntimeException to be thrown when an instance of the class EWarningGeneral have any trouble
    with the dates at the time of its creation.
    """

    def __int__(self, message):
        """
        Basic constructor that calls to super constructor of Exception.

        :param string message: Exception message error to be shown.
        """
        super().__init__(message)
